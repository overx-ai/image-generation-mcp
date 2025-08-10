#!/usr/bin/env python3
"""
Test the text image generation functionality with real MCP API calls.
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

async def test_text_generation():
    """Test text image generation with different models."""
    print("🎨 Testing Text Image Generation")
    print("=" * 80)
    
    async with create_server_session() as session:
        
        # Check available tools
        tools = await session.list_tools()
        tool_names = [tool.name for tool in tools.tools]
        print(f"📋 Available tools: {', '.join(tool_names)}")
        
        # Test cases for text generation
        test_cases = [
            {
                "name": "Ideogram V3 Turbo - Infographic with Text",
                "tool": "generate_text_image",
                "params": {
                    "prompt": 'Infographic showing "Free Time = Better Life" with icons representing different OverX AI tools: currency converter, website blocker, AI assistants, language learning bots. Modern flat design with OverX AI branding colors, clear typography and visual hierarchy.',
                    "aspect_ratio": "1:1",
                    "model": "ideogram-ai/ideogram-v3-turbo"
                },
                "expected_model": "ideogram-ai/ideogram-v3-turbo"
            },
            {
                "name": "Seedream-3 - Cinematic Text Logo",
                "tool": "generate_text_image",
                "params": {
                    "prompt": 'A cinematic, photorealistic medium shot of a modern tech startup office wall with large stylized text "SEEDREAM 3.0" in metallic chrome letters, with soft golden hour lighting creating lens flares. Professional corporate aesthetic.',
                    "aspect_ratio": "16:9",
                    "model": "bytedance/seedream-3"
                },
                "expected_model": "bytedance/seedream-3"
            },
            {
                "name": "Recraft V3 - Logo Design",
                "tool": "generate_text_image",
                "params": {
                    "prompt": 'Clean, professional logo design featuring the text "TechCorp" in a modern sans-serif font, with a simple geometric icon, on a white background. Minimalist design suitable for business cards and letterheads.',
                    "aspect_ratio": "1:1",
                    "model": "recraft-ai/recraft-v3"
                },
                "expected_model": "recraft-ai/recraft-v3"
            }
        ]
        
        results = {}
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 Test {i}: {test_case['name']}")
            print(f"   Tool: {test_case['tool']}")
            print(f"   Model: {test_case['params']['model']}")
            print(f"   Aspect Ratio: {test_case['params']['aspect_ratio']}")
            print(f"   Prompt: {test_case['params']['prompt'][:100]}...")
            
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
                
                # Validate model
                model_ok = actual_model == test_case['expected_model']
                print(f"      {'✅' if model_ok else '⚠️'} Model match: {model_ok}")
                
                results[test_case['name']] = True
                
            except Exception as e:
                print(f"   ❌ Test failed with exception: {e}")
                results[test_case['name']] = False
        
        # Summary
        print(f"\n" + "=" * 80)
        print("📊 TEXT IMAGE GENERATION TEST RESULTS")
        print("=" * 80)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name[:60]:60} {status}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TEXT GENERATION TESTS PASSED!")
            print("\n💡 TEXT GENERATION RECOMMENDATIONS:")
            print("   • Use Ideogram V3 Turbo for infographics and text-heavy images")
            print("   • Use Seedream-3 for cinematic text and premium branding")
            print("   • Use Recraft V3 for logos and clean design work")
            print("   • Include text content in quotes for best results")
            print("   • Text generation excels at: signs, logos, infographics, banners")
            return True
        else:
            print("❌ Some text generation tests failed. Check the output above.")
            return False

if __name__ == "__main__":
    success = asyncio.run(test_text_generation())
    exit(0 if success else 1)