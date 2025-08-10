#!/usr/bin/env python3
"""
MCP Image Generation Server

A FastMCP server that provides AI image generation capabilities using Replicate API
with sophisticated caching, retry logic, and provider abstraction.
"""

import asyncio
import json
import logging
import os
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

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
from core.cache import ImageCache
from providers.replicate import ReplicateProvider

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


async def download_and_process_image(url: str, filepath: Path) -> Tuple[bool, Dict[str, Any]]:
    """
    Download an image from URL, detect format, convert if needed, and save locally.
    
    Args:
        url: The URL of the image to download
        filepath: The local filepath to save the image (extension may be adjusted)
        
    Returns:
        Tuple of (success: bool, metadata: dict with actual_path, format, dimensions)
    """
    logger.info(f"Downloading image from URL: {url}")
    
    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Download image data
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Open image with Pillow to detect format and get dimensions
        image = Image.open(requests.get(url, stream=True).raw)
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
            
        # Save the image in the correct format
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
            
        file_size = actual_filepath.stat().st_size
        logger.info(f"Successfully saved image to: {actual_filepath} ({file_size:,} bytes)")
        
        metadata = {
            'actual_path': actual_filepath,
            'original_format': actual_format,
            'final_format': final_format,
            'dimensions': actual_dimensions,
            'file_size_bytes': file_size
        }
        
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


def generate_local_path(base_path: str, organize_by_date: bool = True) -> Path:
    """Generate a local file path for storing images."""
    base = Path(base_path)
    
    if organize_by_date:
        now = datetime.now()
        base = base / now.strftime("%Y-%m") / now.strftime("%d")
    
    # Generate unique filename
    image_id = str(uuid.uuid4())[:8]
    filename = f"image_{image_id}.png"
    
    return base / filename


async def generate_image(ctx, config, prompt, style, aspect_ratio, model, cache_key, CACHE, replicate_aspect_ratio=None):
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
    # Get API token from environment
    provider_name = config.get("provider", "replicate")
    
    api_token = replicate_api_token
    if not api_token:
        logger.error(f"{provider_name.title()} API token not configured")
        raise ValueError(f"{provider_name.title()} API token not configured. Please set REPLICATE_API_TOKEN in your .env file or environment.")
        
    # Determine model to use
    selected_model = model or config["models"]["style_mapping"].get(style, config["models"]["default"])
    logger.info(f"Using model: {selected_model}")
    
    print(f"Generating image with prompt: '{prompt}' using style: '{style}'")
    print("This may take a few moments...")
    
    # Configure input based on model's aspect ratio capabilities
    from providers.replicate import ReplicateProvider
    
    # Check if model supports native aspect ratios or needs width/height
    model_config = ReplicateProvider.MODEL_CONFIGS.get(selected_model, {})
    aspect_ratio_mode = model_config.get("aspect_ratio_mode", "width_height")
    
    if aspect_ratio_mode == "native":
        # FLUX models - use native aspect_ratio parameter
        model_input = {
            "prompt": prompt,
            "aspect_ratio": replicate_aspect_ratio or "1:1",
            "megapixels": "1"
        }
        logger.info(f"Using native aspect ratio mode for {selected_model}: {replicate_aspect_ratio}")
    else:
        # SDXL models - use width/height parameters
        dimensions = get_aspect_ratio_dimensions(aspect_ratio)
        model_input = {
            "prompt": prompt,
            "width": dimensions["width"],
            "height": dimensions["height"]
        }
        logger.info(f"Using width/height mode for {selected_model}: {dimensions['width']}x{dimensions['height']}")
    
    logger.info(f"Model input configuration: {model_input}")
    
    # Initialize provider based on configuration
    if provider_name == "replicate":
        # Initialize Replicate provider with retry logic
        provider = ReplicateProvider(api_token, config)
    else:
        logger.error(f"Unsupported provider: {provider_name}")
        raise ValueError(f"Unsupported provider: {provider_name}. Currently only 'replicate' is supported.")
    
    # Generate image using provider (handles retry logic internally)
    logger.info(f"Using {provider.__class__.__name__} to generate image")
    
    # Pass all model_input parameters to provider
    generation_params = {
        "prompt": prompt,
        "model": selected_model,
        "ctx": ctx,
    }
    
    # Add model-specific parameters
    for key, value in model_input.items():
        if key != "prompt":  # Don't duplicate prompt
            generation_params[key] = value
    
    generation_result = await provider.generate(**generation_params)
    
    image_url = generation_result["image_url"]
    logger.info(f"Image generation completed, URL: {image_url}")
    
    # Generate local file path with absolute base_path
    base_path = config["storage"]["base_path"]
    if not os.path.isabs(base_path):
        base_path = script_dir / base_path
    
    local_path = generate_local_path(
        str(base_path), 
        config["storage"]["organize_by_date"]
    )
    logger.info(f"Generated local path for image: {local_path}")
    
    # Download and process image to local storage
    download_success, image_metadata = await download_and_process_image(image_url, local_path)
    
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
        "remote_url": image_url,
        "generation_time": generation_result.get("generation_time", 0),
        "attempts": generation_result.get("attempts", 1),
        "requested_aspect_ratio": model_input.get("aspect_ratio", "1:1"),
        "requested_megapixels": model_input.get("megapixels", "1"),
        "actual_dimensions": {
            "width": actual_dimensions[0],
            "height": actual_dimensions[1]
        },
        "original_format": image_metadata.get('original_format', 'unknown'),
        "final_format": image_metadata.get('final_format', 'png'),
        "file_size_bytes": image_metadata.get('file_size_bytes', 0)
    }
    
    # Save metadata alongside image (use same base name as actual image file)
    metadata_path = actual_local_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to: {metadata_path}")
    
    # Save to cache if enabled
    if CACHE and cache_key:
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
        "cache_key": cache_key,
        "metadata_path": str(metadata_path.absolute()),
        "remote_url": image_url,
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
    model: str = None
) -> Dict[str, Any]:
    """
    Generate an image for blog posts using AI image generation with retry logic.
    Intended to use SDXL (budget) by default - excellent quality at ~400 images per $1.
    Currently uses FLUX Schnell (~333 images per $1) as working alternative.
    
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
        style: The style of image (photorealistic, photorealistic_with_text, artistic, cartoon, sketch)
               - photorealistic_with_text: Automatically uses Ideogram V3 Turbo for best text rendering
        model: Optional model override. Options:
               BUDGET (various aspect ratio support):
               - "stability-ai/sdxl" (intended default) - $0.0025/image (~400 per $1) 
               - "black-forest-labs/flux-schnell" (current default) - $0.003/image (~333 per $1)
               PREMIUM (native aspect ratios, higher quality):
               - "black-forest-labs/flux-dev" - $0.030/image (~33 per $1)
               - "black-forest-labs/flux-1.1-pro" - $0.040/image (~25 per $1)
               - "recraft-ai/recraft-v3" - $0.040/image (~25 per $1)
               - "ideogram-ai/ideogram-v3-turbo" - $0.035/image (~28 per $1) [BEST FOR TEXT]
               - "bytedance/seedream-3" - $0.050/image (~20 per $1) [ULTRA PREMIUM]
        
    Returns:
        Dict containing image_url, prompt, model, style, and cost information
    """
    logger.info(f"Starting image generation request - Prompt: '{prompt}', Style: '{style}', Aspect Ratio: '{aspect_ratio}', Model: '{model}'")
    
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
            megapixels="1"
        )
        
        cached_result = CACHE.check_cache(cache_key)
        if cached_result:
            logger.info(f"Cache hit! Returning cached result for key: {cache_key}")
            # Get dimensions from cached metadata
            cached_dimensions = cached_result.get("actual_dimensions", {"width": 1024, "height": 1024})
            dimensions_tuple = (cached_dimensions.get("width", 1024), cached_dimensions.get("height", 1024))
            
            return {
                "image_url": cached_result["image_path"],
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
        result = await generate_image(mock_ctx, config, prompt, style, aspect_ratio, model, cache_key, CACHE, replicate_aspect_ratio)
        
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
    model: str = "black-forest-labs/flux-dev",
    style: str = "photorealistic"
) -> Dict[str, Any]:
    """
    Generate premium quality images using FLUX models with native aspect ratio support.
    Higher cost but superior quality, text rendering, and prompt adherence.
    
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
        model: Premium model to use:
               - "black-forest-labs/flux-dev" (default) - $0.030/image, best quality/prompt adherence  
               - "black-forest-labs/flux-1.1-pro" - $0.040/image, ultra premium quality
               - "recraft-ai/recraft-v3" - $0.040/image, specialized for design/logos
               - "ideogram-ai/ideogram-v3-turbo" - $0.035/image, BEST for images with text
               - "bytedance/seedream-3" - $0.050/image, ultra-premium cinematic quality
        style: The style of image (photorealistic, photorealistic_with_text, artistic, cartoon, sketch)
               - photorealistic_with_text: Automatically uses Ideogram V3 Turbo for text rendering
        
    Returns:
        Dict containing image_url, prompt, model, style, cost and quality information
    """
    logger.info(f"Starting PREMIUM image generation - Prompt: '{prompt}', Model: '{model}', Aspect Ratio: '{aspect_ratio}'")
    
    # Validate premium model (flux-schnell moved to budget)
    premium_models = [
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
    
    # Check cache first if enabled
    cache_key = None
    if CACHE:
        cache_key = CACHE.generate_cache_key(
            prompt=prompt,
            model=model,
            style=style,
            aspect_ratio=aspect_ratio,
            replicate_aspect_ratio=replicate_aspect_ratio,
            megapixels="1"
        )
        
        cached_result = CACHE.check_cache(cache_key)
        if cached_result:
            logger.info(f"Cache hit! Returning cached result for key: {cache_key}")
            # Get dimensions from cached metadata
            cached_dimensions = cached_result.get("actual_dimensions", {"width": 1024, "height": 1024})
            dimensions_tuple = (cached_dimensions.get("width", 1024), cached_dimensions.get("height", 1024))
            
            return {
                "image_url": cached_result["image_path"],
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
        result = await generate_image(mock_ctx, config, prompt, style, aspect_ratio, model, cache_key, CACHE, replicate_aspect_ratio)
        
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
    model: str = "ideogram-ai/ideogram-v3-turbo"
) -> Dict[str, Any]:
    """
    Generate images with text using specialized text-rendering models.
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
        style: The style of image (photorealistic, photorealistic_with_text, artistic, cartoon, sketch)
               - photorealistic_with_text: Automatically uses Ideogram V3 Turbo for text rendering
        model: Text-specialized model to use:
               - "ideogram-ai/ideogram-v3-turbo" (default) - $0.035/image, best for text rendering
               - "bytedance/seedream-3" - $0.050/image, cinematic quality with text support
               - "recraft-ai/recraft-v3" - $0.040/image, design/logo specialist
        
    Returns:
        Dict containing image_url, prompt, model, style, cost and text quality information
    """
    logger.info(f"Starting TEXT image generation - Prompt: '{prompt}', Model: '{model}', Aspect Ratio: '{aspect_ratio}'")
    
    # Validate text-specialized model
    text_models = [
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
    
    # Check cache first if enabled
    cache_key = None
    if CACHE:
        cache_key = CACHE.generate_cache_key(
            prompt=prompt,
            model=model,
            style=style,
            aspect_ratio=aspect_ratio,
            replicate_aspect_ratio=replicate_aspect_ratio,
            megapixels="1"
        )
        
        cached_result = CACHE.check_cache(cache_key)
        if cached_result:
            logger.info(f"Cache hit! Returning cached result for key: {cache_key}")
            # Get dimensions from cached metadata
            cached_dimensions = cached_result.get("actual_dimensions", {"width": 1024, "height": 1024})
            dimensions_tuple = (cached_dimensions.get("width", 1024), cached_dimensions.get("height", 1024))
            
            return {
                "image_url": cached_result["image_path"],
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
        result = await generate_image(mock_ctx, config, prompt, style, aspect_ratio, model, cache_key, CACHE, replicate_aspect_ratio)
        
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
        logger.info("Available tools: generate_blog_image, generate_premium_image, generate_text_image, list_generated_images, get_cache_stats, clear_cache")
        
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")