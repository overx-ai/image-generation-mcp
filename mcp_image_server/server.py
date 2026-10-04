#!/usr/bin/env python3
"""
MCP Image Generation Server

A FastMCP server that provides AI image generation capabilities using Replicate API
with sophisticated caching, retry logic, and provider abstraction.
"""

import asyncio
import hashlib
import io
import json
import logging
import os
import random
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import replicate
import requests
import yaml
from dotenv import load_dotenv
from fastmcp import FastMCP
from PIL import Image

# Load environment variables from .env file relative to script directory
script_dir = Path(__file__).parent.parent
env_file = script_dir / '.env'
load_dotenv(dotenv_path=env_file)

# Import our core components
from mcp_image_server.core.cache import ImageCache
from mcp_image_server.providers.base import BaseProvider
from mcp_image_server.providers.openai_images import (
    DEFAULT_OUTPUT_FORMAT as OPENAI_DEFAULT_OUTPUT_FORMAT,
    DEFAULT_QUALITY as OPENAI_DEFAULT_QUALITY,
    DEFAULT_SIZE as OPENAI_DEFAULT_SIZE,
    OpenAIImagesProvider,
)
from mcp_image_server.providers.replicate import ReplicateProvider
from mcp_image_server.utils.image_optimizer import ImageOptimizer

MODEL_CONFIGS = {**ReplicateProvider.MODEL_CONFIGS, **OpenAIImagesProvider.MODEL_CONFIGS}

REMOTE_URL_SCHEMES = ("http://", "https://")


def is_remote_url(ref: str) -> bool:
    return ref.startswith(REMOTE_URL_SCHEMES)


def is_openai_direct(model: str) -> bool:
    return model in OpenAIImagesProvider.MODEL_CONFIGS

# Initialize MCP server 
mcp = FastMCP("Image Generation Server")

# Configure comprehensive logging
log_file = script_dir / 'image_server.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get API token from environment
replicate_api_token = os.environ.get("REPLICATE_API_TOKEN")
if not replicate_api_token:
    logger.warning("REPLICATE_API_TOKEN environment variable not found.")
    print("⚠️ Warning: REPLICATE_API_TOKEN environment variable not set.")
    print("   Please set it in your .env file or environment: REPLICATE_API_TOKEN=your_api_key")

# OpenAI key: required for gpt-image-2.5 (OpenAI-direct), optional for gpt-image-2
# (passed through Replicate; without it gpt-image-2 is Replicate-billed).
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key:
    logger.info("OPENAI_API_KEY detected - gpt-image-2.5 and gpt-image-2 will bill your OpenAI account")
else:
    logger.warning("OPENAI_API_KEY not set - gpt-image-2.5 is unavailable, falling back to Replicate models")


def load_config(config_path: str = 'config/config.yaml') -> dict:
    """
    Load configuration from config.yaml file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        # Make path relative to this script's directory
        if not os.path.isabs(config_path):
            config_path = Path(__file__).parent / config_path
        else:
            config_path = Path(config_path)
            
        logger.info(f"Loading configuration from: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info("YAML configuration loaded successfully")
            return config
            
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found. Please create config.yaml.")
        print(f"Configuration file {config_path} not found. Please create config.yaml.")
        exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing {config_path}: {e}. Please check YAML syntax.")
        print(f"Error parsing {config_path}: {e}. Please check YAML syntax.")
        exit(1)


# Load global configuration
CONFIG = load_config()

# Initialize cache system
cache_config = CONFIG.get("cache", {})
if cache_config.get("enabled", True):
    # Make cache_dir absolute if it's relative
    cache_dir = cache_config.get("cache_dir", "./cache")
    if not os.path.isabs(cache_dir):
        cache_dir = script_dir / cache_dir
    
    CACHE = ImageCache(
        cache_dir=str(cache_dir),
        ttl_days=cache_config.get("ttl_days", 30),
        max_size_mb=cache_config.get("max_size_mb", 1000)
    )
    logger.info("Cache system initialized")
else:
    CACHE = None
    logger.info("Cache system disabled")


async def download_and_process_image(url: str, filepath: Path, optimize_for_web: bool = True) -> Tuple[bool, Dict[str, Any]]:
    """
    Download an image from URL, detect format, convert if needed, optimize for web, and save locally.
    
    Args:
        url: The URL of the image to download
        filepath: The local filepath to save the image (extension may be adjusted)
        optimize_for_web: Whether to optimize the image for web use
        
    Returns:
        Tuple of (success: bool, metadata: dict with actual_path, format, dimensions, optimization info)
    """
    logger.info(f"Downloading image from URL: {url}")
    
    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if is_remote_url(url):
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            image_bytes = response.content
        else:
            image_bytes = Path(url).read_bytes()

        # Open image with Pillow to detect format and get dimensions
        image = Image.open(io.BytesIO(image_bytes))
        actual_format = image.format.lower() if image.format else 'unknown'
        actual_dimensions = image.size  # (width, height)
        
        logger.info(f"Image format detected: {actual_format}, dimensions: {actual_dimensions}")
        
        # Determine correct file extension based on actual format
        format_extensions = {
            'jpeg': '.jpg',
            'jpg': '.jpg', 
            'png': '.png',
            'webp': '.webp',
            'gif': '.gif',
            'bmp': '.bmp'
        }
        
        correct_extension = format_extensions.get(actual_format, '.png')
        
        # Adjust filepath if extension doesn't match format
        if filepath.suffix.lower() != correct_extension:
            actual_filepath = filepath.with_suffix(correct_extension)
            logger.info(f"Adjusting file extension from {filepath.suffix} to {correct_extension}")
        else:
            actual_filepath = filepath
            
        # Save the image initially
        # Convert WebP and other formats to PNG for better compatibility
        if actual_format in ['webp', 'bmp'] or correct_extension not in ['.jpg', '.png']:
            # Convert to PNG for better compatibility
            actual_filepath = actual_filepath.with_suffix('.png')
            image = image.convert('RGB') if image.mode in ['RGBA', 'P'] else image
            image.save(str(actual_filepath), 'PNG', optimize=True)
            logger.info(f"Converted {actual_format} to PNG format")
            final_format = 'png'
        else:
            # Save in original format if it's already JPG or PNG
            if actual_format == 'jpeg' and image.mode in ['RGBA', 'P']:
                image = image.convert('RGB')
            image.save(str(actual_filepath), image.format, optimize=True, quality=90)
            final_format = actual_format
        
        # Close the image to free resources before optimization
        image.close()
        
        original_size = actual_filepath.stat().st_size
        optimization_result = None
        
        # Optimize for web if requested
        if optimize_for_web:
            logger.info(f"Optimizing image for web deployment")
            
            # Determine target format based on content
            # Keep original format, don't convert PNG to JPEG
            target_format = None
            
            optimization_result = ImageOptimizer.optimize_for_web(
                input_path=actual_filepath,
                output_path=None,  # Optimize in place
                format=target_format,
                create_webp=True,  # Always create WebP variant
                create_responsive=False  # Can be enabled via config later
            )
            
            if optimization_result.get('success'):
                logger.info(
                    f"Web optimization successful: {original_size:,} -> {optimization_result['optimized_size']:,} bytes "
                    f"({optimization_result['size_reduction']}% reduction)"
                )
                
                # Update final format if it changed during optimization
                with Image.open(actual_filepath) as img:
                    final_format = img.format.lower() if img.format else final_format
                    actual_dimensions = img.size
            else:
                logger.warning(f"Web optimization failed: {optimization_result.get('error')}")
        
        file_size = actual_filepath.stat().st_size
        logger.info(f"Successfully saved image to: {actual_filepath} ({file_size:,} bytes)")
        
        metadata = {
            'actual_path': actual_filepath,
            'original_format': actual_format,
            'final_format': final_format,
            'dimensions': actual_dimensions,
            'file_size_bytes': file_size,
            'original_size_bytes': original_size,
            'optimized': optimize_for_web and optimization_result and optimization_result.get('success', False)
        }
        
        # Add optimization details if available
        if optimization_result:
            metadata['optimization'] = optimization_result
        
        return True, metadata
        
    except requests.RequestException as e:
        logger.error(f"Error downloading image from {url}: {e}")
        return False, {}
    except Exception as e:
        logger.error(f"Error processing image from {url}: {e}")
        return False, {}


def get_model_for_style(style: str) -> str:
    """Get the appropriate model based on style."""
    model = CONFIG["models"]["style_mapping"].get(style, CONFIG["models"]["default"])
    logger.info(f"Selected model '{model}' for style '{style}'")
    return model


# Fallback models are configured by family; the concrete model is picked to match the
# quality the caller asked gpt-image-2 for, so a fallback is not a quality downgrade.
NANO_BANANA_BY_QUALITY = {
    "low": "google/nano-banana-2",
    "medium": "google/nano-banana-2",
    "high": "google/nano-banana-pro",
    "auto": "google/nano-banana-pro",
}


def resolve_fallback_model(name: str, quality: str) -> str:
    """Resolve a config fallback entry to a concrete model id."""
    if name == "nano-banana":
        return NANO_BANANA_BY_QUALITY.get(quality or "medium", "google/nano-banana-2")
    return name


def clamp_aspect_ratio(ratio: str, supported) -> str:
    """Map a ratio onto the nearest value a model actually accepts.

    Models disagree on their aspect_ratio enums (seedream-3 has no 4:5, nano-banana
    has no 9:21), so a fallback must not fail on a ratio the primary model allowed.
    """
    if not supported or ratio in supported:
        return ratio

    def value(r: str) -> Optional[float]:
        try:
            w, h = r.split(":")
            return float(w) / float(h)
        except (ValueError, ZeroDivisionError):
            return None

    target = value(ratio)
    candidates = [(value(r), r) for r in supported]
    candidates = [(v, r) for v, r in candidates if v is not None]
    if target is None or not candidates:
        return "1:1" if "1:1" in supported else supported[0]
    nearest = min(candidates, key=lambda c: abs(c[0] - target))[1]
    logger.info(f"Aspect ratio '{ratio}' unsupported by model, using nearest '{nearest}'")
    return nearest


def get_aspect_ratio_dimensions(aspect_ratio: str) -> Dict[str, int]:
    """Convert aspect ratio to width/height for SDXL models."""
    aspect_ratio_dimensions = {
        "landscape": {"width": 1216, "height": 832},   # ~16:9, optimized for SDXL
        "16:9": {"width": 1216, "height": 832},
        "portrait": {"width": 832, "height": 1216},    # ~9:16
        "9:16": {"width": 832, "height": 1216},
        "square": {"width": 1024, "height": 1024},     # 1:1
        "1:1": {"width": 1024, "height": 1024},
        "4:3": {"width": 1152, "height": 896},         # Classic
        "3:4": {"width": 896, "height": 1152},
        "21:9": {"width": 1344, "height": 576},        # Ultra-wide
        "9:21": {"width": 576, "height": 1344},
        "3:2": {"width": 1152, "height": 768},         # Photography
        "2:3": {"width": 768, "height": 1152},
        "5:4": {"width": 1280, "height": 1024},        # Classic monitor
        "4:5": {"width": 1024, "height": 1280}
    }
    return aspect_ratio_dimensions.get(aspect_ratio, {"width": 1024, "height": 1024})


# gpt-image-2 only accepts a restricted set of aspect ratios / fixed sizes and
# rejects ratios like 16:9. Map every ratio our tools accept onto a value from
# gpt-image-2's enum (ratio strings or fixed pixel sizes).
GPT_IMAGE_ASPECT_MAP = {
    "square": "1:1", "1:1": "1:1",
    "landscape": "3:2", "3:2": "3:2", "4:3": "3:2", "5:4": "3:2",
    "16:9": "2048x1152", "21:9": "2048x1152",
    "portrait": "2:3", "2:3": "2:3", "3:4": "2:3", "4:5": "2:3",
    "9:16": "2160x3840", "9:21": "2160x3840",
    "auto": "auto",
}


def resolve_gpt_aspect_ratio(aspect_ratio: str) -> str:
    """Map a user/tool aspect ratio onto gpt-image-2's allowed values."""
    return GPT_IMAGE_ASPECT_MAP.get(aspect_ratio, "1:1")


# The OpenAI Images API takes WIDTHxHEIGHT (both divisible by 16), not ratios.
OPENAI_SIZE_MAP = {
    "square": "1024x1024", "1:1": "1024x1024",
    "landscape": "1536x1024", "3:2": "1536x1024", "4:3": "1536x1024", "5:4": "1536x1024",
    "16:9": "2048x1152", "21:9": "2048x1152",
    "portrait": "1024x1536", "2:3": "1024x1536", "3:4": "1024x1536", "4:5": "1024x1536",
    "9:16": "1152x2048", "9:21": "1152x2048",
    "auto": "auto",
}


def resolve_openai_size(aspect_ratio: str) -> str:
    """Map a user/tool aspect ratio onto an OpenAI Images API size."""
    return OPENAI_SIZE_MAP.get(aspect_ratio, OPENAI_DEFAULT_SIZE)


# Quality variants for gpt-image-2 (drives price). Ignored by models without
# per-quality pricing, so it is safe to accept on every generation tool.
VALID_QUALITIES = ("low", "medium", "high", "auto")


def validate_quality(quality: str) -> Optional[Dict[str, Any]]:
    """Return an error dict if quality is unsupported, else None."""
    if quality not in VALID_QUALITIES:
        return {
            "error": f"Invalid quality '{quality}'. Supported: {', '.join(VALID_QUALITIES)}",
            "error_type": "ValidationError",
            "valid_options": list(VALID_QUALITIES),
        }
    return None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def generate_local_path(base_path: str, organize_by_date: bool = True, filename: str = None) -> Path:
    """Generate a local file path for storing images.

    If `filename` is given it is used verbatim (a `.png` extension is appended
    when it lacks a recognized image extension); otherwise a unique
    `image_<uuid>.png` name is generated.
    """
    base = Path(base_path)

    if organize_by_date:
        now = datetime.now()
        base = base / now.strftime("%Y-%m") / now.strftime("%d")

    if filename:
        if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
            filename = f"{filename}.png"
    else:
        filename = f"image_{str(uuid.uuid4())[:8]}.png"

    return base / filename


def resolve_output_base(config: Dict[str, Any], output_dir: str = None) -> Tuple[Path, bool]:
    """Resolve where images should be saved.

    When `output_dir` is given, images go straight there (no YYYY-MM/DD
    subfolders). Relative paths resolve against the server's working directory —
    prefer absolute paths. Otherwise the config-driven base_path is used.
    Returns (base_path, organize_by_date).
    """
    if output_dir:
        base = Path(output_dir).expanduser()
        if not base.is_absolute():
            base = base.resolve()
        return base, False

    base = Path(config["storage"]["base_path"])
    if not base.is_absolute():
        base = script_dir / base
    return base, config["storage"]["organize_by_date"]


def resolve_reference_images(reference_image: Union[str, List[str], None]) -> Tuple[list, Optional[str]]:
    """Normalize reference image input and produce a stable cache identity.

    Accepts a single str (local path or http(s) URL) or a list of them. Local
    paths are validated and returned as Path objects — the replicate client
    uploads them directly, so no manual upload/base64 is needed. Returns
    (items, identity) where items feeds the model input and identity is a
    stable string for the cache key (None when no reference is given).
    """
    if not reference_image:
        return [], None

    refs = [reference_image] if isinstance(reference_image, str) else list(reference_image)
    items = []
    identity_parts = []
    for ref in refs:
        if is_remote_url(ref):
            items.append(ref)
            identity_parts.append(ref)
        else:
            path = Path(ref).expanduser()
            if not path.is_file():
                raise ValueError(f"Reference image not found: {ref}")
            items.append(path)
            # ponytail: sha256 over file bytes is fine for image-sized files;
            # switch to path+mtime if references ever get large.
            identity_parts.append(hashlib.sha256(path.read_bytes()).hexdigest())

    return items, "|".join(identity_parts)


def save_cached_to_output(src_path: str, config: Dict[str, Any], output_dir: str = None,
                          filename: str = None) -> str:
    """Copy a cache-hit image to a requested output location; return the new path."""
    base, organize = resolve_output_base(config, output_dir)
    if organize:
        now = datetime.now()
        base = base / now.strftime("%Y-%m") / now.strftime("%d")

    if filename:
        name = filename
        if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
            name = f"{name}{Path(src_path).suffix}"
    else:
        name = Path(src_path).name

    dest = base / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest)
    return str(dest.resolve())


def demo() -> None:
    """Self-check for reference/output helpers. Run: python -m mcp_image_server.server --demo"""
    import tempfile

    # str -> list, URL passthrough, missing path raises
    assert resolve_reference_images(None) == ([], None)
    items, ident = resolve_reference_images("https://example.com/a.png")
    assert items == ["https://example.com/a.png"] and ident == "https://example.com/a.png"
    try:
        resolve_reference_images("/no/such/file.png")
        assert False, "expected ValueError for missing file"
    except ValueError:
        pass

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ref.png"
        p.write_bytes(b"fake-image-bytes")
        items, ident = resolve_reference_images(str(p))
        assert items == [p]
        assert ident == resolve_reference_images(str(p))[1]  # stable hash

        # filename appends default extension; custom output dir skips date folders
        cfg = {"storage": {"base_path": d, "organize_by_date": True}}
        base, organize = resolve_output_base(cfg, str(p.parent / "out"))
        assert organize is False and base.is_absolute()
        lp = generate_local_path(str(base), organize, filename="hero")
        assert lp.name == "hero.png" and lp.parent == base

    print("demo: OK")
    _demo_fallback()
    print("demo(fallback): OK")
    _demo_async_jobs()
    print("demo(async jobs): OK")


def _demo_fallback() -> None:
    """Self-check for the model fallback chain and per-model input building (no API)."""
    from mcp_image_server.providers.replicate import ReplicateProvider
    from mcp_image_server.providers.base import is_non_retryable

    # Quality-matched fallback tier
    assert resolve_fallback_model("nano-banana", "medium") == "google/nano-banana-2"
    assert resolve_fallback_model("nano-banana", "high") == "google/nano-banana-pro"
    assert resolve_fallback_model("bytedance/seedream-3", "high") == "bytedance/seedream-3"

    # Aspect ratios are clamped to what each model accepts
    seedream = ReplicateProvider.MODEL_CONFIGS["bytedance/seedream-3"]["supported_ratios"]
    assert clamp_aspect_ratio("16:9", seedream) == "16:9"          # passthrough
    assert clamp_aspect_ratio("4:5", seedream) == "3:4"            # nearest to 0.8
    assert clamp_aspect_ratio("9:21", seedream) == "9:16"          # nearest to 0.43

    # Attempt chain: gpt-image-2 -> same model w/o our key -> nano-banana -> seedream
    cfg = {"models": {"fallback": ["nano-banana", "bytedance/seedream-3"]}}
    cfg25 = {"models": {"fallback": ["openai/gpt-image-2", "nano-banana", "bytedance/seedream-3"]}}
    flare = "openai/gpt-image-2.5-flare"
    global openai_api_key
    saved_key, openai_api_key = openai_api_key, "sk-test"
    try:
        chain = build_attempts(cfg, "openai/gpt-image-2", "medium")
        assert chain[0] == ("openai/gpt-image-2", True)
        assert chain[1] == ("openai/gpt-image-2", False)
        assert chain[2] == ("google/nano-banana-2", False)
        assert chain[3] == ("bytedance/seedream-3", False)

        # Reference images drop models that cannot take them (seedream-3)
        ref_chain = build_attempts(cfg, "openai/gpt-image-2", "medium", ["https://x/a.png"])
        models = [m for m, _ in ref_chain]
        assert "bytedance/seedream-3" not in models
        assert "google/nano-banana-2" in models

        # Model inputs are built per model config
        gpt = build_model_input("openai/gpt-image-2", "p", "1:1", "1:1", "medium")
        assert gpt["openai_api_key"] == "sk-test" and "megapixels" not in gpt
        gpt_no_key = build_model_input("openai/gpt-image-2", "p", "1:1", "1:1", "medium",
                                       use_openai_key=False)
        assert "openai_api_key" not in gpt_no_key

        # gpt-image-2.5 (OpenAI-direct) -> gpt-image-2 Replicate-billed -> nano-banana -> seedream
        chain25 = build_attempts(cfg25, flare, "medium")
        assert chain25 == [(flare, True), ("openai/gpt-image-2", False),
                           ("google/nano-banana-2", False), ("bytedance/seedream-3", False)], chain25
        g25 = build_model_input(flare, "p", "16:9", "16:9", "high")
        assert g25 == {"prompt": "p", "size": "2048x1152", "quality": "high", "output_format": "png"}, g25
        g25_ref = build_model_input(flare, "p", "1:1", "1:1", None, reference_images=["https://x/a.png"])
        assert g25_ref["image"] == ["https://x/a.png"] and g25_ref["quality"] == "medium"
        assert all(int(n) % 16 == 0 for s in OPENAI_SIZE_MAP.values() if s != "auto" for n in s.split("x"))

        # Without an OpenAI key, gpt-image-2.5 is skipped rather than failing every call
        openai_api_key = None
        assert build_attempts(cfg25, flare, "medium")[0] == ("openai/gpt-image-2", False)
        # ...unless nothing else is left: kept so get_provider names the missing key
        assert build_attempts({}, flare, "medium") == [(flare, True)]
        try:
            get_provider(flare, CONFIG)
            raise AssertionError("get_provider must reject OpenAI-direct models without a key")
        except ValueError as missing_key:
            assert "OPENAI_API_KEY" in str(missing_key)
    finally:
        openai_api_key = saved_key

    nano = build_model_input("google/nano-banana-2", "p", "16:9", "16:9", "medium")
    assert nano["resolution"] == "2K" and "megapixels" not in nano
    assert nano["output_format"] == "png"  # model default is jpg
    assert "openai_api_key" not in nano
    flux = build_model_input("black-forest-labs/flux-dev", "p", "16:9", "16:9", "medium")
    assert flux["megapixels"] == "1"

    # Billing/auth errors must not burn the retry budget
    assert is_non_retryable(Exception("ModelError: Your token does not have sufficient credits. (E001)"))
    assert is_non_retryable(Exception("ModelError: Your API key is invalid. (E002)"))
    assert not is_non_retryable(Exception("connection reset by peer"))

    # OpenAI-direct request path (requests.post mocked): references go to /images/edits,
    # API errors stay recognisable as non-retryable and are length-capped.
    from unittest import mock
    from mcp_image_server.providers import openai_images
    provider = OpenAIImagesProvider("sk-test", CONFIG)
    ok = mock.Mock(ok=True, json=lambda: {"data": [{"b64_json": "aGk="}]})
    with mock.patch.object(openai_images.requests, "post", return_value=ok) as post, \
            mock.patch.object(openai_images, "_reference_file", return_value=("a.png", b"x", "image/png")):
        assert provider._request({"model": "m"}, ["https://x/a.png"]) == b"hi"
        assert post.call_args.args[0].endswith("/images/edits") and post.call_args.kwargs["files"][0][0] == "image[]"
        provider._request({"model": "m"}, None)
        assert post.call_args.args[0].endswith("/images/generations") and "json" in post.call_args.kwargs
    quota = mock.Mock(ok=False, status_code=429,
                      json=lambda: {"error": {"code": "insufficient_quota", "message": "x" * 5000}})
    with mock.patch.object(openai_images.requests, "post", return_value=quota):
        try:
            provider._request({"model": "m"}, None)
            raise AssertionError("an API error must raise")
        except Exception as api_error:
            assert is_non_retryable(api_error) and len(str(api_error)) < 1000, str(api_error)[:100]

    # URL references: oversized downloads are refused; a nameless URL gets the default name
    with mock.patch.object(openai_images, "REFERENCE_MAX_BYTES", 2), \
            mock.patch.object(openai_images.requests, "get") as get:
        download = get.return_value.__enter__.return_value.raw.read
        download.return_value = b"abc"
        try:
            openai_images._reference_file("https://x/big.png")
            raise AssertionError("an oversized reference must be refused")
        except ValueError:
            pass
        download.return_value = b"ab"
        assert openai_images._reference_file("https://x/?v=1") == ("reference.png", b"ab", "image/png")


def _demo_async_jobs() -> None:
    """Self-check for the async start/poll job lifecycle (stubs generation, no API)."""
    class _Stub:
        def __init__(self, fn):
            self.fn = fn

    async def checks():
        global generate_premium_image
        orig = generate_premium_image
        try:
            # success path
            async def ok_fn(**kw):
                await asyncio.sleep(0.2)
                return {"image_url": "/tmp/x.png", "prompt": kw.get("prompt"), "model": kw.get("model")}
            generate_premium_image = _Stub(ok_fn)
            started = await start_image_generation.fn(prompt="p", aspect_ratio="1:1")
            assert started["status"] == "processing" and started["job_id"], started
            jid = started["job_id"]
            assert (await get_image_result.fn(jid))["status"] == "processing"
            await asyncio.sleep(0.4)
            done = await get_image_result.fn(jid)
            assert done["status"] == "completed" and done["image_url"] == "/tmp/x.png", done
            assert (await get_image_result.fn(jid))["status"] == "unknown"  # returned once

            # exception path
            async def boom_fn(**kw):
                await asyncio.sleep(0.1)
                raise RuntimeError("kaboom")
            generate_premium_image = _Stub(boom_fn)
            s2 = await start_image_generation.fn(prompt="p2", aspect_ratio="1:1")
            await asyncio.sleep(0.3)
            f = await get_image_result.fn(s2["job_id"])
            assert f["status"] == "failed" and "kaboom" in f["error"], f

            # error-dict path (tool returns error dict instead of raising)
            async def errdict_fn(**kw):
                await asyncio.sleep(0.05)
                return {"error": "bad", "error_type": "ValidationError"}
            generate_premium_image = _Stub(errdict_fn)
            s3 = await start_image_generation.fn(prompt="p3", aspect_ratio="1:1")
            await asyncio.sleep(0.2)
            e = await get_image_result.fn(s3["job_id"])
            assert e["status"] == "failed" and e.get("error") == "bad", e
        finally:
            generate_premium_image = orig

    asyncio.run(checks())


def build_model_input(model, prompt, aspect_ratio, replicate_aspect_ratio=None, quality=None,
                      output_format=None, reference_images=None, use_openai_key=True):
    """Build the provider input dict for a model, per its MODEL_CONFIGS entry."""
    model_config = MODEL_CONFIGS.get(model, {})
    aspect_ratio_mode = model_config.get("aspect_ratio_mode", "width_height")

    if aspect_ratio_mode == "openai_size":
        # gpt-image-2.5 via the OpenAI Images API - WxH size, key held by the provider
        model_input = {
            "prompt": prompt,
            "size": resolve_openai_size(aspect_ratio),
            "quality": quality or model_config.get("default_quality", OPENAI_DEFAULT_QUALITY),
            "output_format": output_format or OPENAI_DEFAULT_OUTPUT_FORMAT,
        }
        logger.info(f"Using OpenAI size mode for {model}: size={model_input['size']}, "
                    f"quality={model_input['quality']}")
    elif aspect_ratio_mode == "gpt_image":
        # gpt-image-2 - restricted aspect ratios + quality variant, NO megapixels
        gpt_aspect_ratio = resolve_gpt_aspect_ratio(aspect_ratio)
        model_input = {
            "prompt": prompt,
            "aspect_ratio": gpt_aspect_ratio,
            "quality": quality or model_config.get("default_quality", "medium"),
            "output_format": output_format or "png",
        }
        # Use our own OpenAI key when configured (billed to that account directly).
        # use_openai_key=False falls back to Replicate's own billing.
        if openai_api_key and use_openai_key:
            model_input["openai_api_key"] = openai_api_key
        logger.info(
            f"Using gpt-image mode for {model}: "
            f"aspect_ratio={gpt_aspect_ratio}, quality={model_input['quality']}, "
            f"own_openai_key={'yes' if openai_api_key and use_openai_key else 'no'}"
        )
    elif aspect_ratio_mode == "native":
        # Models with a native aspect_ratio input (FLUX, nano-banana, seedream, ...)
        model_input = {
            "prompt": prompt,
            "aspect_ratio": clamp_aspect_ratio(replicate_aspect_ratio or "1:1",
                                               model_config.get("supported_ratios")),
        }
        if model_config.get("supports_megapixels"):  # FLUX-only input
            model_input["megapixels"] = "1"
        if model_config.get("supports_output_format"):  # nano-banana defaults to jpg
            model_input["output_format"] = output_format or "png"
        model_input.update(model_config.get("extra_input", {}))
        logger.info(f"Using native aspect ratio mode for {model}: {model_input['aspect_ratio']}")
    else:
        # SDXL models - use width/height parameters
        dimensions = get_aspect_ratio_dimensions(aspect_ratio)
        model_input = {
            "prompt": prompt,
            "width": dimensions["width"],
            "height": dimensions["height"]
        }
        logger.info(f"Using width/height mode for {model}: {dimensions['width']}x{dimensions['height']}")

    # Inject reference image(s) when provided and supported by the selected model
    if reference_images:
        image_param = model_config.get("image_input_param")
        if not image_param:
            supported = sorted(
                m for m, c in MODEL_CONFIGS.items()
                if c.get("image_input_param")
            )
            raise ValueError(
                f"Model '{model}' does not support reference images. "
                f"Supported models: {', '.join(supported)}"
            )
        model_input[image_param] = (
            reference_images if model_config.get("image_input_is_list") else reference_images[0]
        )
        logger.info(f"Added {len(reference_images)} reference image(s) as '{image_param}'")

    loggable_input = {k: ("***redacted***" if k == "openai_api_key" else v) for k, v in model_input.items()}
    logger.info(f"Model input configuration for {model}: {loggable_input}")
    return model_input


def build_attempts(config, selected_model, quality, reference_images=None):
    """Ordered (model, use_openai_key) attempts: the requested model first, then fallbacks.

    Rung 2 re-runs gpt-image-2 without our OpenAI key (Replicate-billed) - that alone
    recovers from "your token does not have sufficient credits". Rung 3+ switch models.
    OpenAI-direct models (gpt-image-2.5) are skipped when no OpenAI key is configured.
    """
    attempts = [(selected_model, True)]
    if openai_api_key and MODEL_CONFIGS.get(selected_model, {}).get("aspect_ratio_mode") == "gpt_image":
        attempts.append((selected_model, False))
    for name in config.get("models", {}).get("fallback", []):
        candidate = resolve_fallback_model(name, quality)
        if candidate != selected_model:
            attempts.append((candidate, False))

    if reference_images:
        usable = [a for a in attempts if MODEL_CONFIGS.get(a[0], {}).get("image_input_param")]
        # Empty means the requested model itself can't take references - keep it so
        # build_model_input raises the explanatory ValueError instead of nothing.
        attempts = usable or attempts[:1]

    if not openai_api_key:
        keyless = [a for a in attempts if not is_openai_direct(a[0])]
        # An all-OpenAI-direct chain is kept so get_provider raises the missing-key error.
        if keyless and keyless != attempts:
            skipped = [m for m, _ in attempts if is_openai_direct(m)]
            logger.warning(f"Skipping {skipped}: OPENAI_API_KEY not set")
            attempts = keyless
    return attempts


def get_provider(model: str, config: Dict[str, Any]) -> BaseProvider:
    """The provider that serves `model`, with its credential."""
    if is_openai_direct(model):
        if not openai_api_key:
            raise ValueError(f"{model} needs OPENAI_API_KEY in .env")
        return OpenAIImagesProvider(openai_api_key, config)
    if not replicate_api_token:
        raise ValueError("Replicate API token not configured. Please set REPLICATE_API_TOKEN "
                         "in your .env file or environment.")
    return ReplicateProvider(replicate_api_token, config)


async def generate_image(ctx, config, prompt, style, aspect_ratio, model, cache_key, CACHE, replicate_aspect_ratio=None, quality=None, output_format=None, reference_images=None, output_dir=None, filename=None):
    """
    Core image generation function with retry logic handled by provider.
    
    Args:
        ctx: MCP context for logging
        config: Configuration dictionary
        prompt: Text description of the image to generate
        style: Style of image
        aspect_ratio: Aspect ratio
        model: Model to use
        cache_key: Cache key for storing result
        CACHE: Cache instance
        
    Returns:
        Dict containing generation results
    """
    # Determine model to use
    selected_model = model or config["models"]["style_mapping"].get(style, config["models"]["default"])
    logger.info(f"Using model: {selected_model}")
    
    print(f"Generating image with prompt: '{prompt}' using style: '{style}'")
    print("This may take a few moments...")
    
    # Each attempt runs on its model's provider (per-model retry logic lives there),
    # walking the fallback chain when a model fails outright.
    requested_model = selected_model
    attempts = build_attempts(config, selected_model, quality, reference_images)
    generation_result = None

    for index, (candidate, use_openai_key) in enumerate(attempts):
        model_input = build_model_input(
            candidate, prompt, aspect_ratio, replicate_aspect_ratio, quality,
            output_format, reference_images, use_openai_key=use_openai_key,
        )
        generation_params = {"prompt": prompt, "model": candidate, "ctx": ctx}
        generation_params.update({k: v for k, v in model_input.items() if k != "prompt"})

        try:
            provider = get_provider(candidate, config)
            generation_result = await provider.generate(**generation_params)
            selected_model = candidate
            break
        except Exception as attempt_error:
            if index == len(attempts) - 1:
                raise
            next_model, next_key = attempts[index + 1]
            logger.warning(
                f"Model '{candidate}' (own_openai_key={'yes' if use_openai_key else 'no'}) "
                f"failed: {attempt_error}. Falling back to '{next_model}' "
                f"(own_openai_key={'yes' if next_key else 'no'})"
            )

    fallback_from = requested_model if selected_model != requested_model else None
    if fallback_from:
        logger.info(f"Generated with fallback model '{selected_model}' instead of '{fallback_from}'")

    image_url = generation_result["image_url"]
    remote_url = image_url if is_remote_url(image_url) else None
    logger.info(f"Image generation completed, URL: {image_url}")

    try:
        # Resolve where to save (custom output_dir/filename or config-driven default)
        base_path, organize_by_date = resolve_output_base(config, output_dir)
        local_path = generate_local_path(str(base_path), organize_by_date, filename=filename)
        logger.info(f"Generated local path for image: {local_path}")

        # Download and process image to local storage with web optimization
        web_optimize = CONFIG.get("optimization", {}).get("enabled", True)
        logger.info(f"Web optimization enabled: {web_optimize}")
        download_success, image_metadata = await download_and_process_image(image_url, local_path, optimize_for_web=web_optimize)
    finally:
        if not remote_url:
            Path(image_url).unlink(missing_ok=True)  # OpenAI-direct temp file
    
    if not download_success:
        logger.error(f"Failed to download image to local path: {local_path}")
        raise RuntimeError("Failed to download and save image locally")
    
    # Use actual path from image processing (may have different extension)
    actual_local_path = image_metadata.get('actual_path', local_path)
    actual_dimensions = image_metadata.get('dimensions', (1024, 1024))  # Default fallback dimensions
    
    # Create comprehensive metadata
    metadata = {
        "prompt": prompt,
        "style": style,
        "aspect_ratio": aspect_ratio,
        "model": selected_model,
        "generated_at": datetime.now().isoformat(),
        "remote_url": remote_url,
        "generation_time": generation_result.get("generation_time", 0),
        "attempts": generation_result.get("attempts", 1),
        "requested_aspect_ratio": model_input.get("aspect_ratio") or model_input.get("size", "1:1"),
        "requested_megapixels": model_input.get("megapixels", "1"),
        "requested_quality": model_input.get("quality"),
        "fallback_from": fallback_from,
        "actual_dimensions": {
            "width": actual_dimensions[0],
            "height": actual_dimensions[1]
        },
        "original_format": image_metadata.get('original_format', 'unknown'),
        "final_format": image_metadata.get('final_format', 'png'),
        "file_size_bytes": image_metadata.get('file_size_bytes', 0),
        "original_size_bytes": image_metadata.get('original_size_bytes', 0),
        "optimized": image_metadata.get('optimized', False),
        "optimization_details": image_metadata.get('optimization', {})
    }
    
    # Save metadata alongside image (use same base name as actual image file)
    metadata_path = actual_local_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to: {metadata_path}")
    
    # A fallback result is not cached under the requested model's key, or every
    # later call would keep returning it instead of retrying the requested model.
    if CACHE and cache_key and not fallback_from:
        try:
            with open(actual_local_path, 'rb') as f:
                image_data = f.read()
            
            cache_metadata = metadata.copy()
            cache_metadata.update({
                "estimated_cost": generation_result.get("estimated_cost", 0.025),
                "metadata_path": str(metadata_path.absolute())
            })
            
            if CACHE.save_to_cache(cache_key, image_data, cache_metadata):
                logger.info(f"Successfully saved result to cache with key: {cache_key}")
            else:
                logger.warning(f"Failed to save result to cache with key: {cache_key}")
                
        except Exception as cache_error:
            logger.error(f"Error saving to cache: {str(cache_error)}", exc_info=True)
    
    print(f"✓ Image generated and saved to: {actual_local_path}")
    
    result = {
        "image_url": str(actual_local_path.absolute()),
        "prompt": prompt,
        "model": selected_model,
        "style": style,
        "aspect_ratio": aspect_ratio,
        "cost": generation_result.get("estimated_cost", 0.025),
        "cached": False,
        "fallback_from": fallback_from,
        "cache_key": cache_key,
        "metadata_path": str(metadata_path.absolute()),
        "remote_url": remote_url,
        "generation_time": generation_result.get("generation_time", 0),
        "attempts": generation_result.get("attempts", 1),
        "dimensions": actual_dimensions,
        "format": image_metadata.get('final_format', 'png'),
        "original_format": image_metadata.get('original_format', 'unknown'),
        "file_size_bytes": image_metadata.get('file_size_bytes', 0)
    }
    logger.info(f"Image generation completed successfully. Result: {result}")
    return result


@mcp.tool()
async def generate_blog_image(
    prompt: str,
    aspect_ratio: str,
    style: str = "photorealistic",
    model: str = None,
    quality: str = "medium",
    output_dir: str = None,
    filename: str = None,
    reference_image: Union[str, List[str]] = None
) -> Dict[str, Any]:
    """
    Generate an image for blog posts using AI image generation with retry logic.
    Defaults to OpenAI gpt-image-2.5 Flare at "medium" quality (~$0.013/image) -
    the recommended setting. Override `model` for Sunburst, FLUX or SDXL.
    If it fails (e.g. the OpenAI account is out of credit) the server falls back
    to gpt-image-2 on Replicate's billing and then to a quality-matched
    fallback model; `fallback_from` in the result names the original model.

    Args:
        prompt: The text description of the image to generate
        aspect_ratio: REQUIRED - The aspect ratio for the image. Supports:
                     USER-FRIENDLY: landscape (16:9), portrait (9:16), square (1:1)
                     PRECISE RATIOS: 1:1, 16:9, 21:9, 3:2, 2:3, 4:5, 5:4, 3:4, 4:3, 9:16, 9:21
                     RECOMMENDED FOR BLOGS:
                     - 16:9 (landscape): Wide format, perfect for blog headers
                     - 4:3 (classic): Good balance for blog content
                     - 1:1 (square): Social media sharing
                     - 9:16 (portrait): Mobile-first content
                     NOTE: gpt-image models support 1:1, 3:2, 2:3 and a few fixed sizes;
                     other ratios are mapped to the closest supported value.
        style: The style of image (photorealistic, photorealistic_with_text, artistic, cartoon, sketch)
               - photorealistic_with_text: Automatically uses Ideogram V3 Turbo for best text rendering
        model: Optional model override. Options:
               - "openai/gpt-image-2.5-flare" (default) - fast, quality-priced: low $0.006 / medium $0.013 / high $0.053
               - "openai/gpt-image-2.5-sunburst" - same price, slower, finer detail and tighter edits
               - "openai/gpt-image-2" - previous gen via Replicate: low $0.012 / medium $0.047 / high $0.128
               BUDGET (various aspect ratio support):
               - "stability-ai/sdxl" - $0.0025/image (~400 per $1)
               - "black-forest-labs/flux-schnell" - $0.003/image (~333 per $1)
               PREMIUM (native aspect ratios, higher quality):
               - "black-forest-labs/flux-dev" - $0.030/image (~33 per $1)
               - "black-forest-labs/flux-1.1-pro" - $0.040/image (~25 per $1)
               - "recraft-ai/recraft-v3" - $0.040/image (~25 per $1)
               - "ideogram-ai/ideogram-v3-turbo" - $0.035/image (~28 per $1) [BEST FOR TEXT]
               - "bytedance/seedream-3" - $0.050/image (~20 per $1) [ULTRA PREMIUM]
               - "google/nano-banana-2" - ~$0.06/image, 2K, reference images [FALLBACK for low/medium]
               - "google/nano-banana-pro" - ~$0.139/image, 2K, reference images [FALLBACK for high/auto]
        quality: gpt-image quality variant - "medium" (default and recommended),
                 "low", "high", or "auto". Prices per model above.
                 Ignored by non-gpt models.
        output_dir: Optional directory to save the image in. When set, the image
                    is saved directly here (no YYYY-MM/DD subfolders). Prefer an
                    absolute path; relative paths resolve against the server cwd.
        filename: Optional output filename (e.g. "hero.png"). ".png" is appended
                  when no image extension is given. Auto-named when omitted.
        reference_image: Optional local path or http(s) URL (or a list of them)
                         used as a visual reference / image-to-image seed. Only
                         gpt-image-2.5 (the default), gpt-image-2 and nano-banana
                         support this; other
                         models return an error.

    Returns:
        Dict containing image_url, prompt, model, style, and cost information
    """
    logger.info(f"Starting image generation request - Prompt: '{prompt}', Style: '{style}', Aspect Ratio: '{aspect_ratio}', Model: '{model}', Quality: '{quality}'")

    quality_error = validate_quality(quality)
    if quality_error:
        return quality_error
    
    # Validate required aspect_ratio parameter using Replicate's format
    # Map user-friendly terms to Replicate ratios for backward compatibility
    aspect_ratio_mapping = {
        "landscape": "16:9",    # Standard widescreen
        "portrait": "9:16",     # Standard mobile/portrait
        "square": "1:1",        # Perfect square
        # Direct Replicate ratios (pass through)
        "1:1": "1:1",
        "16:9": "16:9", 
        "21:9": "21:9",
        "3:2": "3:2",
        "2:3": "2:3", 
        "4:5": "4:5",
        "5:4": "5:4",
        "3:4": "3:4",
        "4:3": "4:3",
        "9:16": "9:16",
        "9:21": "9:21"
    }
    
    if not aspect_ratio:
        error_msg = "aspect_ratio parameter is required. Supported ratios: " + ", ".join(sorted(aspect_ratio_mapping.keys()))
        logger.error(error_msg)
        return {
            "error": error_msg,
            "error_type": "ValidationError",
            "prompt": prompt,
            "valid_options": list(aspect_ratio_mapping.keys())
        }
    
    if aspect_ratio not in aspect_ratio_mapping:
        error_msg = f"Invalid aspect_ratio '{aspect_ratio}'. Supported ratios: {', '.join(sorted(aspect_ratio_mapping.keys()))}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "error_type": "ValidationError", 
            "prompt": prompt,
            "provided_value": aspect_ratio,
            "valid_options": list(aspect_ratio_mapping.keys())
        }
    
    # Get the actual Replicate aspect ratio
    replicate_aspect_ratio = aspect_ratio_mapping[aspect_ratio]
    logger.info(f"Mapped aspect_ratio '{aspect_ratio}' to Replicate format: '{replicate_aspect_ratio}'")
    
    # Load configuration
    config = load_config('config/config.yaml')

    # Resolve reference image(s) up-front (validates paths, feeds the cache key)
    try:
        reference_images, reference_image_id = resolve_reference_images(reference_image)
    except ValueError as ref_err:
        logger.error(str(ref_err))
        return {"error": str(ref_err), "error_type": "ValidationError", "prompt": prompt}
    
    # Check cache first if enabled
    cache_key = None
    if CACHE:
        selected_model_for_cache = model or config["models"]["style_mapping"].get(style, config["models"]["default"])
        cache_key = CACHE.generate_cache_key(
            prompt=prompt,
            model=selected_model_for_cache,
            style=style,
            aspect_ratio=aspect_ratio,
            replicate_aspect_ratio=replicate_aspect_ratio,
            megapixels="1",
            quality=quality,
            reference_image_id=reference_image_id
        )
        
        cached_result = CACHE.check_cache(cache_key)
        if cached_result:
            logger.info(f"Cache hit! Returning cached result for key: {cache_key}")
            # Get dimensions from cached metadata
            cached_dimensions = cached_result.get("actual_dimensions", {"width": 1024, "height": 1024})
            dimensions_tuple = (cached_dimensions.get("width", 1024), cached_dimensions.get("height", 1024))
            
            # Honor a custom output location even on a cache hit
            cached_image_url = cached_result["image_path"]
            if output_dir or filename:
                try:
                    cached_image_url = save_cached_to_output(cached_result["image_path"], config, output_dir, filename)
                except Exception as copy_err:
                    logger.warning(f"Could not copy cached image to output: {copy_err}")
            return {
                "image_url": cached_image_url,
                "prompt": prompt,
                "model": cached_result.get("model", selected_model_for_cache),
                "style": style,
                "aspect_ratio": aspect_ratio,
                "cost": cached_result.get("estimated_cost", 0.0),
                "cached": True,
                "cache_key": cache_key,
                "metadata_path": cached_result.get("metadata_path", ""),
                "remote_url": cached_result.get("remote_url", ""),
                "dimensions": dimensions_tuple,
                "format": cached_result.get("final_format", "png"),
                "original_format": cached_result.get("original_format", "unknown"),
                "file_size_bytes": cached_result.get("file_size_bytes", 0)
            }
        else:
            logger.info(f"Cache miss for key: {cache_key}, proceeding with generation")
    
    # Create a mock context for internal use (FastMCP 2.0 doesn't expose ctx to tools)
    class MockContext:
        def info(self, message): logger.info(f"MCP: {message}")
        def error(self, message): logger.error(f"MCP: {message}")
    
    mock_ctx = MockContext()
    
    try:
        # Generate image using provider's built-in retry logic
        result = await generate_image(mock_ctx, config, prompt, style, aspect_ratio, model, cache_key, CACHE, replicate_aspect_ratio, quality=quality, reference_images=reference_images, output_dir=output_dir, filename=filename)

        success_msg = f"Image generation successful"
        logger.info(success_msg)
        
        return result
        
    except Exception as e:
        error_msg = f"Image generation failed: {str(e)} (Type: {type(e).__name__})"
        logger.error(error_msg)
        
        # Return error information instead of raising exception to maintain MCP tool contract
        return {
            "error": f"Image generation failed: {str(e)}",
            "error_type": type(e).__name__,
            "prompt": prompt,
            "style": style,
            "last_error": str(e)
        }


@mcp.tool()
async def generate_premium_image(
    prompt: str,
    aspect_ratio: str,
    model: str = "openai/gpt-image-2.5-flare",
    style: str = "photorealistic",
    quality: str = "medium",
    output_dir: str = None,
    filename: str = None,
    reference_image: Union[str, List[str]] = None
) -> Dict[str, Any]:
    """
    Generate premium quality images. Defaults to OpenAI gpt-image-2 at "medium"
    quality (the recommended setting; pass quality="high" for the top tier);
    FLUX/Ideogram/Recraft/Seedream/nano-banana models remain selectable via the
    model param. On failure the server falls back automatically (see
    generate_blog_image) and reports the substitution as `fallback_from`.

    Args:
        prompt: The text description of the image to generate
        aspect_ratio: REQUIRED - Native Replicate aspect ratios supported:
                     1:1, 16:9, 21:9, 3:2, 2:3, 4:5, 5:4, 3:4, 4:3, 9:16, 9:21
                     USER-FRIENDLY: landscape→16:9, portrait→9:16, square→1:1
                     PREMIUM FORMATS:
                     - 21:9: Ultra-wide blog headers, perfect for hero sections
                     - 16:9: Standard blog headers, social media
                     - 4:3: Professional content, presentations
                     - 1:1: Social media posts, Instagram
                     NOTE: gpt-image models support 1:1, 3:2, 2:3 and a few fixed sizes;
                     other ratios are mapped to the closest supported value.
        model: Premium model to use:
               - "openai/gpt-image-2.5-flare" (default) - fast, quality-priced: low $0.006 / medium $0.013 / high $0.053
               - "openai/gpt-image-2.5-sunburst" - same price, slower, finer detail and tighter edits
               - "openai/gpt-image-2" - previous gen via Replicate: low $0.012 / medium $0.047 / high $0.128
               - "black-forest-labs/flux-dev" - $0.030/image, best FLUX quality/prompt adherence
               - "black-forest-labs/flux-1.1-pro" - $0.040/image, ultra premium quality
               - "recraft-ai/recraft-v3" - $0.040/image, specialized for design/logos
               - "ideogram-ai/ideogram-v3-turbo" - $0.035/image, BEST for images with text
               - "bytedance/seedream-3" - $0.050/image, ultra-premium cinematic quality
               - "google/nano-banana-2" - ~$0.06/image, 2K, reference images
               - "google/nano-banana-pro" - ~$0.139/image, 2K, reference images
        style: The style of image (photorealistic, photorealistic_with_text, artistic, cartoon, sketch)
               - photorealistic_with_text: Automatically uses Ideogram V3 Turbo for text rendering
        quality: gpt-image quality variant - "medium" (default and recommended),
                 "low", "high", or "auto". Prices per model above.
                 Ignored by non-gpt models.
        output_dir: Optional directory to save the image in. When set, the image
                    is saved directly here (no YYYY-MM/DD subfolders). Prefer an
                    absolute path; relative paths resolve against the server cwd.
        filename: Optional output filename (e.g. "hero.png"). ".png" is appended
                  when no image extension is given. Auto-named when omitted.
        reference_image: Optional local path or http(s) URL (or a list of them)
                         used as a visual reference / image-to-image seed. Only
                         gpt-image-2.5 (the default), gpt-image-2 and nano-banana
                         support this; other
                         models return an error.

    Returns:
        Dict containing image_url, prompt, model, style, cost and quality information
    """
    logger.info(f"Starting PREMIUM image generation - Prompt: '{prompt}', Model: '{model}', Aspect Ratio: '{aspect_ratio}', Quality: '{quality}'")

    # Validate premium model (flux-schnell moved to budget)
    premium_models = [
        "openai/gpt-image-2.5-flare",
        "openai/gpt-image-2.5-sunburst",
        "openai/gpt-image-2",
        "black-forest-labs/flux-dev",
        "black-forest-labs/flux-1.1-pro",
        "recraft-ai/recraft-v3",
        "ideogram-ai/ideogram-v3-turbo",
        "bytedance/seedream-3"
    ]

    if model not in premium_models:
        return {
            "error": f"Invalid premium model '{model}'. Supported models: {', '.join(premium_models)}",
            "error_type": "ValidationError",
            "supported_models": premium_models
        }

    quality_error = validate_quality(quality)
    if quality_error:
        return quality_error
    
    # Validate required aspect_ratio parameter using Replicate's format
    # Map user-friendly terms to Replicate ratios for backward compatibility
    aspect_ratio_mapping = {
        "landscape": "16:9",    # Standard widescreen
        "portrait": "9:16",     # Standard mobile/portrait
        "square": "1:1",        # Perfect square
        # Direct Replicate ratios (pass through)
        "1:1": "1:1",
        "16:9": "16:9", 
        "21:9": "21:9",
        "3:2": "3:2",
        "2:3": "2:3", 
        "4:5": "4:5",
        "5:4": "5:4",
        "3:4": "3:4",
        "4:3": "4:3",
        "9:16": "9:16",
        "9:21": "9:21"
    }
    
    if not aspect_ratio:
        error_msg = "aspect_ratio parameter is required. Supported ratios: " + ", ".join(sorted(aspect_ratio_mapping.keys()))
        logger.error(error_msg)
        return {
            "error": error_msg,
            "error_type": "ValidationError",
            "prompt": prompt,
            "valid_options": list(aspect_ratio_mapping.keys())
        }
    
    if aspect_ratio not in aspect_ratio_mapping:
        error_msg = f"Invalid aspect_ratio '{aspect_ratio}'. Supported ratios: {', '.join(sorted(aspect_ratio_mapping.keys()))}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "error_type": "ValidationError", 
            "prompt": prompt,
            "provided_value": aspect_ratio,
            "valid_options": list(aspect_ratio_mapping.keys())
        }
    
    # Get the actual Replicate aspect ratio
    replicate_aspect_ratio = aspect_ratio_mapping[aspect_ratio]
    logger.info(f"Mapped aspect_ratio '{aspect_ratio}' to Replicate format: '{replicate_aspect_ratio}'")
    
    # Load configuration
    config = load_config('config/config.yaml')

    # Resolve reference image(s) up-front (validates paths, feeds the cache key)
    try:
        reference_images, reference_image_id = resolve_reference_images(reference_image)
    except ValueError as ref_err:
        logger.error(str(ref_err))
        return {"error": str(ref_err), "error_type": "ValidationError", "prompt": prompt}
    
    # Check cache first if enabled
    cache_key = None
    if CACHE:
        cache_key = CACHE.generate_cache_key(
            prompt=prompt,
            model=model,
            style=style,
            aspect_ratio=aspect_ratio,
            replicate_aspect_ratio=replicate_aspect_ratio,
            megapixels="1",
            quality=quality,
            reference_image_id=reference_image_id
        )

        cached_result = CACHE.check_cache(cache_key)
        if cached_result:
            logger.info(f"Cache hit! Returning cached result for key: {cache_key}")
            # Get dimensions from cached metadata
            cached_dimensions = cached_result.get("actual_dimensions", {"width": 1024, "height": 1024})
            dimensions_tuple = (cached_dimensions.get("width", 1024), cached_dimensions.get("height", 1024))

            # Honor a custom output location even on a cache hit
            cached_image_url = cached_result["image_path"]
            if output_dir or filename:
                try:
                    cached_image_url = save_cached_to_output(cached_result["image_path"], config, output_dir, filename)
                except Exception as copy_err:
                    logger.warning(f"Could not copy cached image to output: {copy_err}")
            return {
                "image_url": cached_image_url,
                "prompt": prompt,
                "model": cached_result.get("model", model),
                "style": style,
                "aspect_ratio": aspect_ratio,
                "cost": cached_result.get("estimated_cost", 0.0),
                "cached": True,
                "cache_key": cache_key,
                "metadata_path": cached_result.get("metadata_path", ""),
                "remote_url": cached_result.get("remote_url", ""),
                "dimensions": dimensions_tuple,
                "format": cached_result.get("final_format", "png"),
                "original_format": cached_result.get("original_format", "unknown"),
                "file_size_bytes": cached_result.get("file_size_bytes", 0)
            }
        else:
            logger.info(f"Cache miss for key: {cache_key}, proceeding with generation")

    # Create a mock context for internal use (FastMCP 2.0 doesn't expose ctx to tools)
    class MockContext:
        def info(self, message): logger.info(f"MCP: {message}")
        def error(self, message): logger.error(f"MCP: {message}")

    mock_ctx = MockContext()

    try:
        # Generate image using provider's built-in retry logic
        result = await generate_image(mock_ctx, config, prompt, style, aspect_ratio, model, cache_key, CACHE, replicate_aspect_ratio, quality=quality, reference_images=reference_images, output_dir=output_dir, filename=filename)

        success_msg = f"Premium image generation successful"
        logger.info(success_msg)
        
        return result
        
    except Exception as e:
        error_msg = f"Premium image generation failed: {str(e)} (Type: {type(e).__name__})"
        logger.error(error_msg)
        
        # Return error information instead of raising exception to maintain MCP tool contract
        return {
            "error": f"Premium image generation failed: {str(e)}",
            "error_type": type(e).__name__,
            "prompt": prompt,
            "style": style,
            "last_error": str(e)
        }


@mcp.tool()
async def generate_text_image(
    prompt: str,
    aspect_ratio: str = "16:9",
    style: str = "photorealistic",
    model: str = "openai/gpt-image-2.5-flare",
    quality: str = "medium",
    output_dir: str = None,
    filename: str = None,
    reference_image: Union[str, List[str]] = None
) -> Dict[str, Any]:
    """
    Generate images with text using strong text-rendering models. Defaults to
    OpenAI gpt-image-2.5 Flare at "medium" quality (the recommended setting - text is
    already sharp; only pass quality="high" for dense or very small typography);
    Ideogram/Seedream/Recraft remain selectable via the model param.
    Optimized for images that include readable text, logos, signs, or typography.

    Args:
        prompt: The text description of the image to generate (include text content in quotes)
        aspect_ratio: REQUIRED - Native Replicate aspect ratios supported:
                     1:1, 16:9, 21:9, 3:2, 2:3, 4:5, 5:4, 3:4, 4:3, 9:16, 9:21
                     USER-FRIENDLY: landscape→16:9, portrait→9:16, square→1:1
                     RECOMMENDED FOR TEXT IMAGES:
                     - 16:9: Blog headers with text overlays
                     - 1:1: Social media posts with text
                     - 3:2: Marketing materials, posters
                     - 21:9: Ultra-wide banners with text
                     NOTE: gpt-image models support 1:1, 3:2, 2:3 and a few fixed sizes;
                     other ratios are mapped to the closest supported value.
        style: The style of image (photorealistic, photorealistic_with_text, artistic, cartoon, sketch)
               - photorealistic_with_text: Automatically uses Ideogram V3 Turbo for text rendering
        model: Text-specialized model to use:
               - "openai/gpt-image-2.5-flare" (default) - fast, quality-priced: low $0.006 / medium $0.013 / high $0.053
               - "openai/gpt-image-2.5-sunburst" - same price, slower, finer detail and tighter edits
               - "openai/gpt-image-2" - previous gen via Replicate: low $0.012 / medium $0.047 / high $0.128
               - "ideogram-ai/ideogram-v3-turbo" - $0.035/image, best dedicated text renderer
               - "bytedance/seedream-3" - $0.050/image, cinematic quality with text support
               - "google/nano-banana-2" - ~$0.06/image, 2K, reference images
               - "google/nano-banana-pro" - ~$0.139/image, 2K, reference images
               - "recraft-ai/recraft-v3" - $0.040/image, design/logo specialist
        quality: gpt-image quality variant - "medium" (default and recommended),
                 "low", "high", or "auto". Prices per model above.
                 Ignored by non-gpt models.
        output_dir: Optional directory to save the image in. When set, the image
                    is saved directly here (no YYYY-MM/DD subfolders). Prefer an
                    absolute path; relative paths resolve against the server cwd.
        filename: Optional output filename (e.g. "hero.png"). ".png" is appended
                  when no image extension is given. Auto-named when omitted.
        reference_image: Optional local path or http(s) URL (or a list of them)
                         used as a visual reference / image-to-image seed. Only
                         gpt-image-2.5 (the default), gpt-image-2 and nano-banana
                         support this; other
                         models return an error.

    Returns:
        Dict containing image_url, prompt, model, style, cost and text quality information
    """
    logger.info(f"Starting TEXT image generation - Prompt: '{prompt}', Model: '{model}', Aspect Ratio: '{aspect_ratio}', Quality: '{quality}'")

    # Validate text-specialized model
    text_models = [
        "openai/gpt-image-2.5-flare",
        "openai/gpt-image-2.5-sunburst",
        "openai/gpt-image-2",
        "ideogram-ai/ideogram-v3-turbo",
        "bytedance/seedream-3",
        "recraft-ai/recraft-v3"
    ]

    if model not in text_models:
        return {
            "error": f"Invalid text model '{model}'. Supported models: {', '.join(text_models)}",
            "error_type": "ValidationError",
            "supported_models": text_models
        }

    quality_error = validate_quality(quality)
    if quality_error:
        return quality_error
    
    # Add helpful hint about text in prompts
    if not any(keyword in prompt.lower() for keyword in ['text', 'sign', 'logo', 'writing', 'words', '"']):
        logger.info("Tip: For best text results, include the desired text in quotes in your prompt")
    
    # Validate required aspect_ratio parameter using Replicate's format
    # Map user-friendly terms to Replicate ratios for backward compatibility
    aspect_ratio_mapping = {
        "landscape": "16:9",    # Standard widescreen
        "portrait": "9:16",     # Standard mobile/portrait
        "square": "1:1",        # Perfect square
        # Direct Replicate ratios (pass through)
        "1:1": "1:1",
        "16:9": "16:9", 
        "21:9": "21:9",
        "3:2": "3:2",
        "2:3": "2:3", 
        "4:5": "4:5",
        "5:4": "5:4",
        "3:4": "3:4",
        "4:3": "4:3",
        "9:16": "9:16",
        "9:21": "9:21"
    }
    
    if not aspect_ratio:
        error_msg = "aspect_ratio parameter is required. Supported ratios: " + ", ".join(sorted(aspect_ratio_mapping.keys()))
        logger.error(error_msg)
        return {
            "error": error_msg,
            "error_type": "ValidationError",
            "prompt": prompt,
            "valid_options": list(aspect_ratio_mapping.keys())
        }
    
    if aspect_ratio not in aspect_ratio_mapping:
        error_msg = f"Invalid aspect_ratio '{aspect_ratio}'. Supported ratios: {', '.join(sorted(aspect_ratio_mapping.keys()))}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "error_type": "ValidationError", 
            "prompt": prompt,
            "provided_value": aspect_ratio,
            "valid_options": list(aspect_ratio_mapping.keys())
        }
    
    # Get the actual Replicate aspect ratio
    replicate_aspect_ratio = aspect_ratio_mapping[aspect_ratio]
    logger.info(f"Mapped aspect_ratio '{aspect_ratio}' to Replicate format: '{replicate_aspect_ratio}'")
    
    # Load configuration
    config = load_config('config/config.yaml')

    # Resolve reference image(s) up-front (validates paths, feeds the cache key)
    try:
        reference_images, reference_image_id = resolve_reference_images(reference_image)
    except ValueError as ref_err:
        logger.error(str(ref_err))
        return {"error": str(ref_err), "error_type": "ValidationError", "prompt": prompt}
    
    # Check cache first if enabled
    cache_key = None
    if CACHE:
        cache_key = CACHE.generate_cache_key(
            prompt=prompt,
            model=model,
            style=style,
            aspect_ratio=aspect_ratio,
            replicate_aspect_ratio=replicate_aspect_ratio,
            megapixels="1",
            quality=quality,
            reference_image_id=reference_image_id
        )

        cached_result = CACHE.check_cache(cache_key)
        if cached_result:
            logger.info(f"Cache hit! Returning cached result for key: {cache_key}")
            # Get dimensions from cached metadata
            cached_dimensions = cached_result.get("actual_dimensions", {"width": 1024, "height": 1024})
            dimensions_tuple = (cached_dimensions.get("width", 1024), cached_dimensions.get("height", 1024))

            # Honor a custom output location even on a cache hit
            cached_image_url = cached_result["image_path"]
            if output_dir or filename:
                try:
                    cached_image_url = save_cached_to_output(cached_result["image_path"], config, output_dir, filename)
                except Exception as copy_err:
                    logger.warning(f"Could not copy cached image to output: {copy_err}")
            return {
                "image_url": cached_image_url,
                "prompt": prompt,
                "model": cached_result.get("model", model),
                "style": style,
                "aspect_ratio": aspect_ratio,
                "cost": cached_result.get("estimated_cost", 0.0),
                "cached": True,
                "cache_key": cache_key,
                "metadata_path": cached_result.get("metadata_path", ""),
                "remote_url": cached_result.get("remote_url", ""),
                "dimensions": dimensions_tuple,
                "format": cached_result.get("final_format", "png"),
                "original_format": cached_result.get("original_format", "unknown"),
                "file_size_bytes": cached_result.get("file_size_bytes", 0)
            }
        else:
            logger.info(f"Cache miss for key: {cache_key}, proceeding with generation")

    # Create a mock context for internal use (FastMCP 2.0 doesn't expose ctx to tools)
    class MockContext:
        def info(self, message): logger.info(f"MCP: {message}")
        def error(self, message): logger.error(f"MCP: {message}")

    mock_ctx = MockContext()

    try:
        # Generate image using provider's built-in retry logic
        result = await generate_image(mock_ctx, config, prompt, style, aspect_ratio, model, cache_key, CACHE, replicate_aspect_ratio, quality=quality, reference_images=reference_images, output_dir=output_dir, filename=filename)

        success_msg = f"Text image generation successful"
        logger.info(success_msg)
        
        return result
        
    except Exception as e:
        error_msg = f"Text image generation failed: {str(e)} (Type: {type(e).__name__})"
        logger.error(error_msg)
        
        # Return error information instead of raising exception to maintain MCP tool contract
        return {
            "error": f"Text image generation failed: {str(e)}",
            "error_type": type(e).__name__,
            "prompt": prompt,
            "style": style,
            "last_error": str(e)
        }


# In-memory registry of async generation jobs (job_id -> {task, summary, started_at}).
# ponytail: in-memory — jobs die if the server restarts; persist the Replicate
# prediction id to disk only if mid-render restarts become a real problem.
GENERATION_JOBS: Dict[str, Dict[str, Any]] = {}


@mcp.tool()
async def start_image_generation(
    prompt: str,
    aspect_ratio: str,
    model: str = "openai/gpt-image-2.5-flare",
    style: str = "photorealistic",
    quality: str = "medium",
    output_dir: str = None,
    filename: str = None,
    reference_image: Union[str, List[str]] = None
) -> Dict[str, Any]:
    """
    Start an image generation job in the background and return immediately with a
    job_id. Use this instead of generate_premium_image when the client enforces a
    short tool-call timeout (e.g. Claude Cowork's ~30-60s limit) — a 1-3 minute
    render would otherwise be killed client-side before it finishes.

    After calling this, poll get_image_result(job_id) every ~20-30 seconds until
    status is no longer "processing". The generation runs on the server and is not
    tied to this call's timeout.

    Accepts the same params as generate_premium_image (prompt, aspect_ratio, model,
    style, quality, output_dir, filename, reference_image); see that tool for
    details. `quality` defaults to "medium" (the recommended setting);
    "low", "high" and "auto" are also accepted.
    Returns a dict with status="processing" and a job_id.
    """
    job_id = uuid.uuid4().hex[:12]
    # Reuse the fully-tested premium generation flow verbatim as a background task.
    coro = generate_premium_image.fn(
        prompt=prompt,
        aspect_ratio=aspect_ratio,
        model=model,
        style=style,
        quality=quality,
        output_dir=output_dir,
        filename=filename,
        reference_image=reference_image,
    )
    task = asyncio.create_task(coro)
    GENERATION_JOBS[job_id] = {
        "task": task,
        "summary": prompt[:80],
        "started_at": datetime.now().isoformat(),
    }
    logger.info(f"Started async generation job {job_id} for prompt: '{prompt[:80]}'")
    return {
        "status": "processing",
        "job_id": job_id,
        "prompt": prompt,
        "model": model,
        "poll_with": "get_image_result",
        "hint": "Call get_image_result(job_id) every ~20-30s until status != 'processing'.",
    }


@mcp.tool()
async def get_image_result(job_id: str) -> Dict[str, Any]:
    """
    Poll an async image generation job started with start_image_generation.

    Returns one of:
    - {"status": "processing", ...}  -> not done yet; poll again in ~20-30s
    - {"status": "completed", image_url, prompt, model, ...}  -> done; image saved
    - {"status": "failed", error}  -> generation errored
    - {"status": "unknown", error}  -> no such job_id (never started, or already
                                       retrieved — results are returned once)
    """
    job = GENERATION_JOBS.get(job_id)
    if job is None:
        return {"status": "unknown", "error": f"No such job_id: {job_id}"}

    task = job["task"]
    if not task.done():
        return {"status": "processing", "job_id": job_id, "started_at": job["started_at"]}

    # Job finished — hand back the result once and drop it from the registry.
    GENERATION_JOBS.pop(job_id, None)

    exc = task.exception()
    if exc is not None:
        logger.error(f"Async generation job {job_id} failed: {exc}")
        return {"status": "failed", "error": str(exc), "error_type": type(exc).__name__}

    result = task.result()
    # generate_premium_image returns an error dict (not raising) on failure.
    if isinstance(result, dict) and result.get("error"):
        return {"status": "failed", **result}
    return {"status": "completed", **result}


@mcp.tool()
async def list_generated_images() -> Dict[str, Any]:
    """
    List all generated images in the storage directory.
    
    Returns:
        Dict containing list of generated images with metadata
    """
    logger.info("Listing generated images")
    try:
        # Make base_path absolute if it's relative
        base_path_config = CONFIG["storage"]["base_path"]
        if not os.path.isabs(base_path_config):
            base_path = script_dir / base_path_config
        else:
            base_path = Path(base_path_config)
        logger.info(f"Checking storage path: {base_path}")
        
        if not base_path.exists():
            logger.warning(f"Storage path does not exist: {base_path}")
            return {"images": [], "count": 0}
        
        images = []
        for image_file in base_path.rglob("*.png"):
            metadata_file = image_file.with_suffix('.json')
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            images.append({
                "path": str(image_file.absolute()),
                "filename": image_file.name,
                "size_bytes": image_file.stat().st_size,
                "created": datetime.fromtimestamp(image_file.stat().st_ctime).isoformat(),
                "metadata": metadata
            })
        
        # Sort by creation time, newest first
        images.sort(key=lambda x: x["created"], reverse=True)
        
        logger.info(f"Found {len(images)} images in storage")
        
        return {
            "images": images,
            "count": len(images),
            "storage_path": str(base_path.absolute())
        }
        
    except Exception as e:
        logger.error(f"Error listing images: {str(e)} (Type: {type(e).__name__})", exc_info=True)
        return {
            "error": f"Error listing images: {str(e)}",
            "error_type": type(e).__name__
        }


@mcp.tool()
async def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics and information.
    
    Returns:
        Dict containing cache statistics and configuration
    """
    logger.info("Getting cache statistics")
    
    try:
        if not CACHE:
            logger.info("Cache is disabled")
            return {
                "cache_enabled": False,
                "message": "Cache is disabled"
            }
        
        stats = CACHE.get_cache_stats()
        logger.info(f"Cache stats retrieved: {stats}")
        
        return {
            "cache_enabled": True,
            **stats
        }
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)} (Type: {type(e).__name__})", exc_info=True)
        return {
            "error": f"Error getting cache stats: {str(e)}",
            "error_type": type(e).__name__,
            "cache_enabled": CACHE is not None
        }


@mcp.tool()
async def optimize_existing_image(
    image_path: str,
    create_webp: bool = True,
    create_responsive: bool = False,
    max_width: int = None,
    max_height: int = None
) -> Dict[str, Any]:
    """
    Optimize an existing image for web deployment.
    
    Args:
        image_path: Path to the image to optimize (absolute or relative to storage)
        create_webp: Create a WebP variant for better compression (default: True)
        create_responsive: Create responsive variants for different screen sizes (default: False)
        max_width: Maximum width in pixels (default: 2048)
        max_height: Maximum height in pixels (default: 2048)
        
    Returns:
        Dict containing optimization results including file sizes and variants created
    """
    logger.info(f"Optimizing existing image: {image_path}")
    
    try:
        # Convert to Path object
        input_path = Path(image_path)
        
        # If path is not absolute, check if it's relative to storage directory
        if not input_path.is_absolute():
            storage_base = CONFIG["storage"]["base_path"]
            if not os.path.isabs(storage_base):
                storage_base = script_dir / storage_base
            
            # Try relative to storage directory
            potential_path = storage_base / image_path
            if potential_path.exists():
                input_path = potential_path
            elif not input_path.exists():
                logger.error(f"Image not found: {image_path}")
                return {
                    "error": f"Image not found: {image_path}",
                    "error_type": "FileNotFoundError"
                }
        
        # Check if file exists
        if not input_path.exists():
            logger.error(f"Image file does not exist: {input_path}")
            return {
                "error": f"Image file does not exist: {input_path}",
                "error_type": "FileNotFoundError"
            }
        
        # Get optimization config
        opt_config = CONFIG.get("optimization", {})
        max_width = max_width or opt_config.get("max_width", 2048)
        max_height = max_height or opt_config.get("max_height", 2048)
        
        # Perform optimization
        result = ImageOptimizer.optimize_for_web(
            input_path=input_path,
            output_path=None,  # Optimize in place
            max_width=max_width,
            max_height=max_height,
            create_webp=create_webp,
            create_responsive=create_responsive
        )
        
        if result.get('success'):
            logger.info(
                f"Optimization successful: {result['original_size']:,} -> {result['optimized_size']:,} bytes "
                f"({result['size_reduction']}% reduction)"
            )
            
            # Update metadata file if it exists
            metadata_path = input_path.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                metadata['optimization'] = result
                metadata['optimized_at'] = datetime.now().isoformat()
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Updated metadata file: {metadata_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error optimizing image: {str(e)} (Type: {type(e).__name__})", exc_info=True)
        return {
            "error": f"Error optimizing image: {str(e)}",
            "error_type": type(e).__name__
        }


@mcp.tool()
async def batch_optimize_images(
    directory: str = None,
    create_webp: bool = True,
    create_responsive: bool = False
) -> Dict[str, Any]:
    """
    Batch optimize all images in a directory for web deployment.
    
    Args:
        directory: Directory containing images (default: storage directory)
        create_webp: Create WebP variants for better compression (default: True)
        create_responsive: Create responsive variants for different screen sizes (default: False)
        
    Returns:
        Dict containing batch optimization results and statistics
    """
    logger.info(f"Starting batch optimization for directory: {directory or 'storage'}")
    
    try:
        # Determine directory
        if directory:
            input_dir = Path(directory)
            if not input_dir.is_absolute():
                storage_base = CONFIG["storage"]["base_path"]
                if not os.path.isabs(storage_base):
                    storage_base = script_dir / storage_base
                input_dir = storage_base / directory
        else:
            # Use default storage directory
            storage_base = CONFIG["storage"]["base_path"]
            if not os.path.isabs(storage_base):
                storage_base = script_dir / storage_base
            input_dir = Path(storage_base)
        
        if not input_dir.exists():
            logger.error(f"Directory not found: {input_dir}")
            return {
                "error": f"Directory not found: {input_dir}",
                "error_type": "FileNotFoundError"
            }
        
        # Get optimization config
        opt_config = CONFIG.get("optimization", {})
        
        # Perform batch optimization
        results = ImageOptimizer.batch_optimize(
            input_dir=input_dir,
            output_dir=None,  # Optimize in place
            max_width=opt_config.get("max_width", 2048),
            max_height=opt_config.get("max_height", 2048),
            create_webp=create_webp,
            create_responsive=create_responsive
        )
        
        # Calculate statistics
        total_original = sum(r['original_size'] for r in results if r.get('success'))
        total_optimized = sum(r['optimized_size'] for r in results if r.get('success'))
        successful = [r for r in results if r.get('success')]
        failed = [r for r in results if not r.get('success')]
        
        avg_reduction = 0
        if successful:
            avg_reduction = sum(r['size_reduction'] for r in successful) / len(successful)
        
        stats = {
            "total_images": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "total_original_size": total_original,
            "total_optimized_size": total_optimized,
            "total_saved": total_original - total_optimized,
            "average_reduction": round(avg_reduction, 2),
            "directory": str(input_dir.absolute())
        }
        
        logger.info(
            f"Batch optimization complete: {stats['successful']}/{stats['total_images']} images, "
            f"saved {stats['total_saved']:,} bytes ({stats['average_reduction']}% avg reduction)"
        )
        
        return {
            "success": True,
            "statistics": stats,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in batch optimization: {str(e)} (Type: {type(e).__name__})", exc_info=True)
        return {
            "error": f"Error in batch optimization: {str(e)}",
            "error_type": type(e).__name__
        }


@mcp.tool()
async def clear_cache() -> Dict[str, Any]:
    """
    Clear expired cache entries.
    
    Returns:
        Dict containing information about cleared entries
    """
    logger.info("Clearing expired cache entries")
    
    try:
        if not CACHE:
            logger.warning("Cannot clear cache - cache is disabled")
            return {
                "cache_enabled": False,
                "message": "Cache is disabled"
            }
        
        removed_count = CACHE.clear_expired()
        logger.info(f"Cache cleanup completed - removed {removed_count} expired entries")
        
        return {
            "cache_enabled": True,
            "removed_entries": removed_count,
            "message": f"Successfully removed {removed_count} expired cache entries"
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)} (Type: {type(e).__name__})", exc_info=True)
        return {
            "error": f"Error clearing cache: {str(e)}",
            "error_type": type(e).__name__,
            "cache_enabled": CACHE is not None
        }


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        demo()
        sys.exit(0)
    # Run the FastMCP server
    try:
        # Log startup info to logger instead of stdout to avoid MCP protocol issues
        logger.info("Starting MCP Image Generation Server")
        # Make base_path absolute for logging
        storage_base_path = CONFIG['storage']['base_path']
        if not os.path.isabs(storage_base_path):
            storage_base_path = script_dir / storage_base_path
        logger.info(f"Image storage location: {os.path.abspath(storage_base_path)}")
        
        # Check API token from environment
        api_token = replicate_api_token
        token_source = "environment/env file" if api_token else "none"
        logger.info(f"API Token status: {'Set' if api_token else 'Not set'} ({token_source})")
        logger.info(f"Cache status: {'Enabled' if CACHE else 'Disabled'}")
        logger.info("Available tools: generate_blog_image, generate_premium_image, generate_text_image, start_image_generation, get_image_result, optimize_existing_image, batch_optimize_images, list_generated_images, get_cache_stats, clear_cache")
        
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")