#!/usr/bin/env python3
"""
Test script for image optimization functionality.

This script tests the web optimization features added to the MCP image server:
1. Automatic optimization during image generation
2. Manual optimization of existing images
3. Batch optimization of directories
"""

import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the MCP server components
from mcp_image_server.server import (
    generate_blog_image,
    optimize_existing_image,
    batch_optimize_images,
    list_generated_images
)
from mcp_image_server.utils.image_optimizer import ImageOptimizer


async def test_generation_with_optimization():
    """Test that new image generation includes optimization."""
    print("\n=== Testing Image Generation with Optimization ===")
    
    # Generate a test image
    result = await generate_blog_image(
        prompt="A modern minimalist blog header with abstract geometric shapes",
        aspect_ratio="16:9",
        style="artistic"
    )
    
    if "error" in result:
        print(f"❌ Error generating image: {result['error']}")
        return False
    
    print(f"✓ Image generated: {result['image_url']}")
    print(f"  Original size: {result.get('original_size_bytes', 'N/A'):,} bytes")
    print(f"  Optimized size: {result.get('file_size_bytes', 'N/A'):,} bytes")
    print(f"  Optimized: {result.get('optimized', False)}")
    
    # Check if WebP variant was created
    image_path = Path(result['image_url'])
    webp_path = image_path.with_suffix('.webp')
    
    if webp_path.exists():
        webp_size = webp_path.stat().st_size
        print(f"✓ WebP variant created: {webp_path.name} ({webp_size:,} bytes)")
    else:
        print(f"⚠ WebP variant not found")
    
    # Check optimization details
    if 'optimization_details' in result:
        opt_details = result['optimization_details']
        if opt_details and opt_details.get('success'):
            print(f"✓ Optimization details:")
            print(f"  Size reduction: {opt_details.get('size_reduction', 0)}%")
            if 'webp_variant' in opt_details:
                print(f"  WebP reduction: {opt_details['webp_variant'].get('size_reduction', 0)}%")
    
    return True


async def test_optimize_existing():
    """Test optimization of an existing image."""
    print("\n=== Testing Optimization of Existing Image ===")
    
    # First, get list of images
    images_result = await list_generated_images()
    
    if images_result.get('count', 0) == 0:
        print("⚠ No images found to optimize. Generating one first...")
        
        # Generate an image first
        gen_result = await generate_blog_image(
            prompt="A colorful abstract pattern",
            aspect_ratio="square",
            style="artistic"
        )
        
        if "error" in gen_result:
            print(f"❌ Error generating test image: {gen_result['error']}")
            return False
        
        test_image_path = gen_result['image_url']
    else:
        # Use the first image found
        test_image_path = images_result['images'][0]['path']
    
    print(f"Testing optimization on: {test_image_path}")
    
    # Optimize the image
    opt_result = await optimize_existing_image(
        image_path=test_image_path,
        create_webp=True,
        create_responsive=True  # Test responsive variants
    )
    
    if not opt_result.get('success'):
        print(f"❌ Optimization failed: {opt_result.get('error', 'Unknown error')}")
        return False
    
    print(f"✓ Image optimized successfully")
    print(f"  Original size: {opt_result['original_size']:,} bytes")
    print(f"  Optimized size: {opt_result['optimized_size']:,} bytes")
    print(f"  Size reduction: {opt_result['size_reduction']}%")
    
    # Check for WebP variant
    if 'webp_variant' in opt_result:
        print(f"✓ WebP variant:")
        print(f"  Path: {opt_result['webp_variant']['path']}")
        print(f"  Size reduction: {opt_result['webp_variant']['size_reduction']}%")
    
    # Check for responsive variants
    if opt_result.get('variants'):
        print(f"✓ Responsive variants created: {len(opt_result['variants'])}")
        for variant in opt_result['variants']:
            print(f"  - {variant['width']}w: {variant['size']:,} bytes")
    
    return True


async def test_batch_optimization():
    """Test batch optimization of a directory."""
    print("\n=== Testing Batch Optimization ===")
    
    # Run batch optimization on the storage directory
    batch_result = await batch_optimize_images(
        directory=None,  # Use default storage directory
        create_webp=True,
        create_responsive=False
    )
    
    if "error" in batch_result:
        print(f"❌ Batch optimization failed: {batch_result['error']}")
        return False
    
    stats = batch_result['statistics']
    print(f"✓ Batch optimization completed")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Total saved: {stats['total_saved']:,} bytes")
    print(f"  Average reduction: {stats['average_reduction']}%")
    
    # Show details for first few results
    if batch_result.get('results'):
        print("\nFirst 3 results:")
        for i, result in enumerate(batch_result['results'][:3]):
            if result.get('success'):
                print(f"  {i+1}. {Path(result['input_path']).name}")
                print(f"     Size: {result['original_size']:,} -> {result['optimized_size']:,}")
                print(f"     Reduction: {result['size_reduction']}%")
    
    return True


def test_standalone_optimizer():
    """Test the ImageOptimizer class directly."""
    print("\n=== Testing Standalone ImageOptimizer ===")
    
    # Create a test image if needed
    test_dir = Path("test_images")
    test_dir.mkdir(exist_ok=True)
    
    # Create a simple test image
    from PIL import Image
    test_image_path = test_dir / "test_image.png"
    
    if not test_image_path.exists():
        # Create a test image (1000x1000 with gradient)
        img = Image.new('RGB', (1000, 1000))
        pixels = img.load()
        for i in range(1000):
            for j in range(1000):
                pixels[i, j] = (i % 256, j % 256, (i + j) % 256)
        img.save(test_image_path, 'PNG')
        print(f"Created test image: {test_image_path}")
    
    # Test estimation
    print("\n1. Testing optimization estimation:")
    estimate = ImageOptimizer.estimate_optimization(test_image_path)
    print(f"  Current size: {estimate['current_size']:,} bytes")
    print(f"  Current format: {estimate['current_format']}")
    print(f"  Optimization potential:")
    for format_name, potential in estimate['optimization_potential'].items():
        if 'estimated_reduction' in potential:
            print(f"    {format_name}: ~{potential['estimated_reduction']}% reduction")
    
    # Test actual optimization
    print("\n2. Testing actual optimization:")
    result = ImageOptimizer.optimize_for_web(
        input_path=test_image_path,
        output_path=test_dir / "test_image_optimized.jpg",
        format='jpeg',
        quality=85,
        create_webp=True,
        create_responsive=True
    )
    
    if result['success']:
        print(f"✓ Optimization successful")
        print(f"  Size: {result['original_size']:,} -> {result['optimized_size']:,} bytes")
        print(f"  Reduction: {result['size_reduction']}%")
        
        if 'webp_variant' in result:
            print(f"  WebP variant: {result['webp_variant']['size_reduction']}% reduction")
        
        if result.get('variants'):
            print(f"  Responsive variants: {len(result['variants'])} created")
    else:
        print(f"❌ Optimization failed: {result.get('error')}")
    
    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Image Optimization Test Suite")
    print("=" * 60)
    
    # Check if API token is set
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("\n⚠️ Warning: REPLICATE_API_TOKEN not set.")
        print("   Some tests may fail without a valid API token.")
        print("   Running standalone tests only...")
        
        # Run only standalone tests
        test_standalone_optimizer()
        return
    
    # Run all tests
    tests = [
        ("Generation with Optimization", test_generation_with_optimization),
        ("Optimize Existing Image", test_optimize_existing),
        ("Batch Optimization", test_batch_optimization),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Also run standalone test
    try:
        standalone_success = test_standalone_optimizer()
        results.append(("Standalone Optimizer", standalone_success))
    except Exception as e:
        print(f"\n❌ Standalone test failed with exception: {e}")
        results.append(("Standalone Optimizer", False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️ {total - passed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(main())