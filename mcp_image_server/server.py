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
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import replicate
import yaml
from dotenv import load_dotenv
from fastmcp import FastMCP

# Load environment variables from .env file
load_dotenv()

# Import our core components
from core.cache import ImageCache
from providers.replicate import ReplicateProvider

# Initialize MCP server
mcp = FastMCP("Image Generation Server")

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('image_server.log'),
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
    CACHE = ImageCache(
        cache_dir=cache_config.get("cache_dir", "./cache"),
        ttl_days=cache_config.get("ttl_days", 30),
        max_size_mb=cache_config.get("max_size_mb", 1000)
    )
    logger.info("Cache system initialized")
else:
    CACHE = None
    logger.info("Cache system disabled")


async def download_image(url: str, filepath: Path) -> bool:
    """
    Download an image from URL and save it locally.
    
    Args:
        url: The URL of the image to download
        filepath: The local filepath to save the image
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Downloading image from URL: {url} to path: {filepath}")
    try:
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Download image
        urllib.request.urlretrieve(url, str(filepath))
        logger.info(f"Successfully downloaded image to: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error downloading image from {url}: {e}")
        return False


def get_model_for_style(style: str) -> str:
    """Get the appropriate model based on style."""
    model = CONFIG["models"]["style_mapping"].get(style, CONFIG["models"]["default"])
    logger.info(f"Selected model '{model}' for style '{style}'")
    return model


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


async def generate_image(ctx, config, prompt, style, aspect_ratio, model, cache_key, CACHE):
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
    
    # Configure input based on aspect ratio
    model_input = {"prompt": prompt}
    
    # Add aspect ratio configuration
    if aspect_ratio == "portrait":
        model_input["width"] = 768
        model_input["height"] = 1024
    elif aspect_ratio == "square":
        model_input["width"] = 768
        model_input["height"] = 768
    else:  # landscape
        model_input["width"] = 1024
        model_input["height"] = 768
    
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
    
    generation_result = await provider.generate(
        prompt=prompt,
        model=selected_model,
        ctx=ctx,
        width=model_input.get("width", 1024),
        height=model_input.get("height", 768)
    )
    
    image_url = generation_result["image_url"]
    logger.info(f"Image generation completed, URL: {image_url}")
    
    # Generate local file path
    local_path = generate_local_path(
        config["storage"]["base_path"], 
        config["storage"]["organize_by_date"]
    )
    logger.info(f"Generated local path for image: {local_path}")
    
    # Download image to local storage
    download_success = await download_image(image_url, local_path)
    
    if not download_success:
        logger.error(f"Failed to download image to local path: {local_path}")
        raise RuntimeError("Failed to download and save image locally")
    
    # Create metadata
    metadata = {
        "prompt": prompt,
        "style": style,
        "aspect_ratio": aspect_ratio,
        "model": selected_model,
        "generated_at": datetime.now().isoformat(),
        "remote_url": image_url,
        "generation_time": generation_result.get("generation_time", 0),
        "attempts": generation_result.get("attempts", 1)
    }
    
    # Save metadata alongside image
    metadata_path = local_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to: {metadata_path}")
    
    # Save to cache if enabled
    if CACHE and cache_key:
        try:
            with open(local_path, 'rb') as f:
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
    
    print(f"✓ Image generated and saved to: {local_path}")
    
    result = {
        "image_url": str(local_path.absolute()),
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
        "attempts": generation_result.get("attempts", 1)
    }
    logger.info(f"Image generation completed successfully. Result: {result}")
    return result


@mcp.tool()
async def generate_blog_image(
    ctx,
    prompt: str,
    style: str = "photorealistic", 
    aspect_ratio: str = "landscape",
    model: str = None
) -> Dict[str, Any]:
    """
    Generate an image for blog posts using AI image generation with retry logic.
    
    Args:
        prompt: The text description of the image to generate
        style: The style of image (photorealistic, artistic, cartoon, sketch)
        aspect_ratio: The aspect ratio (landscape, portrait, square)
        model: Optional specific model to use (overrides style-based selection)
        
    Returns:
        Dict containing image_url, prompt, model, style, and cost information
    """
    logger.info(f"Starting image generation request - Prompt: '{prompt}', Style: '{style}', Aspect Ratio: '{aspect_ratio}', Model: '{model}'")
    
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
            width=1024 if aspect_ratio == "landscape" else 768 if aspect_ratio == "portrait" else 768,
            height=768 if aspect_ratio == "landscape" else 1024 if aspect_ratio == "portrait" else 768
        )
        
        cached_result = CACHE.check_cache(cache_key)
        if cached_result:
            logger.info(f"Cache hit! Returning cached result for key: {cache_key}")
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
                "remote_url": cached_result.get("remote_url", "")
            }
        else:
            logger.info(f"Cache miss for key: {cache_key}, proceeding with generation")
    
    try:
        # Generate image using provider's built-in retry logic
        result = await generate_image(ctx, config, prompt, style, aspect_ratio, model, cache_key, CACHE)
        
        success_msg = f"Image generation successful"
        ctx.info(success_msg)
        logger.info(success_msg)
        
        return result
        
    except Exception as e:
        error_msg = f"Image generation failed: {str(e)} (Type: {type(e).__name__})"
        ctx.error(error_msg)
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
async def list_generated_images() -> Dict[str, Any]:
    """
    List all generated images in the storage directory.
    
    Returns:
        Dict containing list of generated images with metadata
    """
    logger.info("Listing generated images")
    try:
        base_path = Path(CONFIG["storage"]["base_path"])
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
        print("🚀 Starting MCP Image Generation Server...")
        print("📁 Image storage location:", os.path.abspath(CONFIG["storage"]["base_path"]))
        
        # Check API token from environment
        api_token = replicate_api_token
        token_source = "environment/env file" if api_token else "none"
        print("🔑 API Token status:", "✓ Set" if api_token else "✗ Not set", f"({token_source})")
        print("\n📘 Available tools:")
        print("  - generate_blog_image: Generate images from text prompts")
        print("  - list_generated_images: List all generated images")
        print("  - get_cache_stats: Get cache statistics and information")
        print("  - clear_cache: Clear expired cache entries")
        print(f"\n🗂️  Cache status: {'✓ Enabled' if CACHE else '✗ Disabled'}")
        print("\nServer starting...")
        mcp.run()
    except KeyboardInterrupt:
        print("\n✗ Server stopped by user")
    except Exception as e:
        print(f"✗ Failed to start server: {e}")