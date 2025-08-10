#!/usr/bin/env python3
"""
Test the new photorealistic_with_text style that automatically uses Ideogram.
"""

import asyncio
import json
import logging
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def create_server_session():
    """Create an MCP client session."""
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "mcp_image_server/server.py"],
        env=None
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session

async def test_photorealistic_with_text_style():
    """Test the new photorealistic_with_text style."""
    print("🎨 Testing photorealistic_with_text Style")
    print("=" * 80)
    
    async with create_server_session() as session:
        
        # Test cases comparing regular vs text styles
        test_cases = [
            {
                "name": "Regular photorealistic style (should use FLUX Schnell)",
                "tool": "generate_blog_image",
                "params": {
                    "prompt": 'Modern office workspace with a computer screen showing "PRODUCTIVITY BOOST" text',
                    "aspect_ratio": "16:9",
                    "style": "photorealistic"
                },
                "expected_model": "black-forest-labs/flux-schnell"
            },
            {
                "name": "photorealistic_with_text style (should use Ideogram)",
                "tool": "generate_blog_image",
                "params": {
                    "prompt": 'Modern office workspace with a computer screen showing "PRODUCTIVITY BOOST" text',
                    "aspect_ratio": "16:9",
                    "style": "photorealistic_with_text"
                },
                "expected_model": "ideogram-ai/ideogram-v3-turbo"
            },
            {
                "name": "photorealistic_with_text - Social Media Post",
                "tool": "generate_blog_image",
                "params": {
                    "prompt": 'Social media post design with bold text "SUCCESS MINDSET" and motivational imagery, clean modern layout',
                    "aspect_ratio": "1:1",
                    "style": "photorealistic_with_text"
                },
                "expected_model": "ideogram-ai/ideogram-v3-turbo"
            }
        ]
        
        results = {}
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 Test {i}: {test_case['name']}")
            print(f"   Tool: {test_case['tool']}")
            print(f"   Style: {test_case['params']['style']}")
            print(f"   Expected Model: {test_case['expected_model']}")
            print(f"   Prompt: {test_case['params']['prompt'][:80]}...")
            
            try:
                result = await session.call_tool(test_case["tool"], test_case["params"])
                
                if result.isError:
                    print(f"   ❌ Tool call failed: {result.content}")
                    results[test_case['name']] = False
                    continue
                
                response_data = json.loads(result.content[0].text)
                
                if "error" in response_data:
                    print(f"   ❌ Generation failed: {response_data['error']}")
                    results[test_case['name']] = False
                    continue
                
                # Analyze results
                actual_cost = response_data.get('cost', 0)
                actual_model = response_data.get('model', 'unknown')
                dimensions = response_data.get('dimensions', [0, 0])
                format_type = response_data.get('format', 'unknown')
                cached = response_data.get('cached', False)
                
                print(f"   ✅ Generation successful!")
                print(f"      Model used: {actual_model}")
                print(f"      Cost: ${actual_cost:.4f}")
                print(f"      Dimensions: {dimensions[0]}x{dimensions[1]}")
                print(f"      Format: {format_type}")
                print(f"      Cached: {cached}")
                print(f"      Image path: {response_data.get('image_url', 'N/A')}")
                
                # Validate model selection
                model_ok = actual_model == test_case['expected_model']
                print(f"      {'✅' if model_ok else '❌'} Model selection: {model_ok}")
                
                if model_ok:
                    print(f"      🎯 Style mapping working correctly!")
                else:
                    print(f"      ⚠️ Expected {test_case['expected_model']}, got {actual_model}")
                
                results[test_case['name']] = model_ok
                
            except Exception as e:
                print(f"   ❌ Test failed with exception: {e}")
                results[test_case['name']] = False
        
        # Summary
        print(f"\n" + "=" * 80)
        print("📊 PHOTOREALISTIC_WITH_TEXT STYLE TEST RESULTS")
        print("=" * 80)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name[:60]:60} {status}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL STYLE MAPPING TESTS PASSED!")
            print("\n💡 STYLE USAGE RECOMMENDATIONS:")
            print("   • Use 'photorealistic' for regular images without text")
            print("   • Use 'photorealistic_with_text' for images with readable text")
            print("   • photorealistic_with_text automatically selects Ideogram V3 Turbo")
            print("   • Perfect for: blog headers, social media posts, infographics")
            print("   • Text will be sharp, readable, and professionally rendered")
            return True
        else:
            print("❌ Some style mapping tests failed. Check the output above.")
            return False

if __name__ == "__main__":
    success = asyncio.run(test_photorealistic_with_text_style())
    exit(0 if success else 1)