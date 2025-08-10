#!/usr/bin/env python3
"""Quick test of image generation fixes"""

import asyncio
from pathlib import Path
from PIL import Image
import sys
import os

# Setup paths
sys.path.insert(0, os.path.join(os.getcwd(), 'mcp_image_server'))

# Import after path setup
import server

async def test_single_image():
    """Generate one image to test fixes"""
    
    # Mock context
    class TestContext:
        def info(self, message): print(f"ℹ️ {message}")
        def error(self, message): print(f"❌ {message}")
    
    ctx = TestContext()
    
    # Test unique prompt to avoid cache
    prompt = "Modern office workspace with laptop showing code editor, coffee cup, notebook - test image for format validation"
    
    print(f"🎨 Testing image generation with new format handling")
    print(f"Prompt: {prompt}")
    
    try:
        # Generate using generate_image function directly
        config = server.load_config()
        
        # Generate cache key
        cache_key = None
        if server.CACHE:
            selected_model = config["models"]["style_mapping"].get("photorealistic", config["models"]["default"])
            cache_key = server.CACHE.generate_cache_key(
                prompt=prompt,
                model=selected_model,
                style="photorealistic",
                aspect_ratio="landscape",
                width=1200,
                height=630
            )
        
        result = await server.generate_image(
            ctx=ctx,
            config=config,
            prompt=prompt,
            style="photorealistic",
            aspect_ratio="landscape",
            model=None,
            cache_key=cache_key,
            CACHE=server.CACHE
        )
        
        # Check if generation was successful
        if "error" in result:
            print(f"❌ Generation failed: {result['error']}")
            return False
        
        image_path = Path(result['image_url'])
        print(f"📁 Generated file: {image_path}")
        print(f"📎 Extension: {image_path.suffix}")
        
        if image_path.exists():
            # Check actual format using Pillow
            with Image.open(image_path) as img:
                print(f"🖼️ Actual format: {img.format}")
                print(f"📐 Dimensions: {img.size}")
            
            # Check file type using system
            import subprocess
            file_output = subprocess.run(['file', str(image_path)], capture_output=True, text=True)
            print(f"🔍 File command output: {file_output.stdout.strip()}")
            
            # Check metadata
            metadata_path = image_path.with_suffix('.json')
            if metadata_path.exists():
                import json
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                print(f"📊 Metadata:")
                print(f"   Original format: {metadata.get('original_format', 'N/A')}")
                print(f"   Final format: {metadata.get('final_format', 'N/A')}")
                if 'actual_dimensions' in metadata:
                    actual_dims = metadata['actual_dimensions']
                    print(f"   Dimensions in metadata: {actual_dims.get('width', 0)}x{actual_dims.get('height', 0)}")
            
            return True
        else:
            print(f"❌ File not found: {image_path}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_single_image())
    print("✅ Test completed successfully!" if success else "❌ Test failed!")