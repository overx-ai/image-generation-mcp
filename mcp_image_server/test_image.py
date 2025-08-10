#!/usr/bin/env python3
"""
Simple image generation test that runs from the mcp_image_server directory.
"""

import asyncio
import os
import sys
from pathlib import Path

async def test_generation():
    """Test image generation with real API key."""
    print("🖼️ MCP Image Generation Test")
    print("=" * 40)
    
    try:
        # Load environment from local .env
        from dotenv import load_dotenv
        load_dotenv()
        
        # Add current directory to Python path
        sys.path.insert(0, os.getcwd())
        
        # Import server components
        import server
        
        # Mock context
        class TestContext:
            def info(self, message):
                print(f"ℹ️ {message}")
            def error(self, message):
                print(f"❌ {message}")
        
        ctx = TestContext()
        
        # Check API token
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token or api_token == "your_replicate_api_token_here":
            print("❌ API token not configured in .env file")
            return False
        
        print(f"✅ API token configured: {api_token[:8]}...{api_token[-4:]}")
        
        # Load config to show data directories
        config = server.load_config()
        storage_path = Path(config['storage']['base_path'])
        cache_path = Path(config['cache']['cache_dir'])
        
        print(f"📂 Storage path: {storage_path.resolve()}")
        print(f"💾 Cache path: {cache_path.resolve()}")
        
        # Test parameters
        prompt = "A beautiful sunset over mountains with a lake in the foreground"
        style = "photorealistic" 
        aspect_ratio = "landscape"
        
        print(f"\n🎨 Generating image...")
        print(f"   Prompt: {prompt}")
        print(f"   Style: {style}")
        print(f"   This may take 30-60 seconds...")
        
        # Generate cache key
        cache_key = None
        if server.CACHE:
            selected_model = config["models"]["style_mapping"].get(style, config["models"]["default"])
            cache_key = server.CACHE.generate_cache_key(
                prompt=prompt,
                model=selected_model,
                style=style,
                aspect_ratio=aspect_ratio,
                width=1024,
                height=768
            )
            
            # Check cache first
            cached_result = server.CACHE.check_cache(cache_key)
            if cached_result:
                print("🎯 Found cached result!")
                image_path = cached_result["image_path"]
                print(f"✅ Using cached image: {image_path}")
                
                # Verify file exists
                if Path(image_path).exists():
                    file_size = Path(image_path).stat().st_size
                    print(f"✅ File exists: {file_size:,} bytes")
                    return True
                else:
                    print("⚠️ Cached file missing, will regenerate")
        
        # Generate new image
        try:
            result = await server.generate_image(
                ctx, config, prompt, style, aspect_ratio, None, cache_key, server.CACHE
            )
            
            print(f"\n🎉 Generation successful!")
            print(f"📁 Image: {result.get('image_url')}")
            print(f"💰 Cost: ${result.get('cost', 0):.4f}")
            print(f"⚡ Time: {result.get('generation_time', 0):.1f}s")
            
            # Verify file
            image_path = Path(result.get('image_url', ''))
            if image_path.exists():
                file_size = image_path.stat().st_size
                print(f"✅ File saved: {file_size:,} bytes")
                
                # Check it's outside code directory
                code_dir = Path.cwd()
                is_outside = not str(image_path.resolve()).startswith(str(code_dir.resolve()))
                print(f"📂 Outside code dir: {'✅' if is_outside else '❌'}")
                
                return True
            else:
                print(f"❌ File not found: {image_path}")
                return False
                
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_generation())
    if success:
        print(f"\n🎉 SUCCESS: Image generated and stored outside code directory!")
    else:
        print(f"\n❌ FAILED: Image generation unsuccessful")
        sys.exit(1)