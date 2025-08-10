#!/usr/bin/env python3
"""
Test the format and dimension fixes.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add mcp_image_server to path
sys.path.insert(0, str(Path(__file__).parent / 'mcp_image_server'))

import server
from PIL import Image

async def test_format_fixes():
    """Test that format and dimension issues are fixed."""
    print("🧪 Testing Format and Dimension Fixes")
    print("=" * 50)
    
    # Mock context
    class TestContext:
        def info(self, message):
            print(f"ℹ️ {message}")
        def error(self, message):
            print(f"❌ {message}")
    
    ctx = TestContext()
    
    try:
        # Load config
        config = server.load_config('mcp_image_server/config/config.yaml')
        
        # Test parameters - use unique prompt to avoid cache
        test_cases = [
            {
                "prompt": "A modern tech workspace with multiple monitors showing code and analytics dashboards",
                "style": "photorealistic",
                "aspect_ratio": "landscape",
                "expected_dimensions": (1200, 630)
            },
            {
                "prompt": "Portrait of a professional software engineer in a modern office environment", 
                "style": "photorealistic",
                "aspect_ratio": "portrait",
                "expected_dimensions": (768, 1024)
            },
            {
                "prompt": "Square format social media image of a developer working on code",
                "style": "photorealistic", 
                "aspect_ratio": "square",
                "expected_dimensions": (1024, 1024)
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🎨 Test Case {i}: {test_case['aspect_ratio']} aspect ratio")
            print(f"   Prompt: {test_case['prompt'][:60]}...")
            print(f"   Expected dimensions: {test_case['expected_dimensions']}")
            
            try:
                # Generate cache key manually to check if cached
                selected_model = config["models"]["style_mapping"].get(test_case['style'], config["models"]["default"])
                cache_key = None
                if server.CACHE:
                    cache_key = server.CACHE.generate_cache_key(
                        prompt=test_case['prompt'],
                        model=selected_model,
                        style=test_case['style'],
                        aspect_ratio=test_case['aspect_ratio'],
                        width=test_case['expected_dimensions'][0],
                        height=test_case['expected_dimensions'][1]
                    )
                
                # Generate image
                result = await server.generate_image(
                    ctx, config, test_case['prompt'], test_case['style'], 
                    test_case['aspect_ratio'], None, cache_key, server.CACHE
                )
                
                image_path = Path(result['image_url'])
                
                # Verify file exists
                if not image_path.exists():
                    print(f"❌ Image file not found: {image_path}")
                    continue
                    
                # Check file extension and format
                print(f"   📁 File path: {image_path}")
                print(f"   📎 File extension: {image_path.suffix}")
                
                # Load and check actual image properties
                with Image.open(image_path) as img:
                    actual_dimensions = img.size
                    actual_format = img.format
                    
                print(f"   🖼️ Actual format: {actual_format}")
                print(f"   📐 Actual dimensions: {actual_dimensions}")
                
                # Check if dimensions match expected
                expected_dims = test_case['expected_dimensions']
                if actual_dimensions == expected_dims:
                    print("   ✅ Dimensions match expected values")
                else:
                    print(f"   ⚠️ Dimension mismatch - Expected: {expected_dims}, Got: {actual_dimensions}")
                
                # Check file extension matches format
                format_extensions = {'PNG': '.png', 'JPEG': '.jpg', 'WEBP': '.webp'}
                expected_ext = format_extensions.get(actual_format, '.png')
                if image_path.suffix.lower() == expected_ext:
                    print("   ✅ File extension matches image format")
                else:
                    print(f"   ⚠️ Extension mismatch - File: {image_path.suffix}, Format: {actual_format}")
                
                # Check metadata
                metadata_path = image_path.with_suffix('.json')
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    
                    requested_dims = metadata.get('requested_dimensions', {})
                    actual_dims = metadata.get('actual_dimensions', {})
                    original_format = metadata.get('original_format', 'unknown')
                    final_format = metadata.get('final_format', 'unknown')
                    
                    print(f"   📊 Metadata - Original format: {original_format}, Final format: {final_format}")
                    print(f"   📊 Metadata - Requested: {requested_dims.get('width', 0)}x{requested_dims.get('height', 0)}")
                    print(f"   📊 Metadata - Actual: {actual_dims.get('width', 0)}x{actual_dims.get('height', 0)}")
                    
                    # Check if metadata matches actual image
                    meta_actual = (actual_dims.get('width', 0), actual_dims.get('height', 0))
                    if meta_actual == actual_dimensions:
                        print("   ✅ Metadata dimensions match actual image")
                    else:
                        print(f"   ⚠️ Metadata dimension mismatch")
                else:
                    print("   ⚠️ Metadata file not found")
                    
                print(f"   ✅ Test case {i} completed")
                
            except Exception as e:
                print(f"   ❌ Test case {i} failed: {e}")
                import traceback
                traceback.print_exc()
                
        print(f"\n✅ Format and dimension testing completed!")
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_format_fixes())