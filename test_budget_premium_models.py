#!/usr/bin/env python3
"""
Test the new budget vs premium model system.
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

async def test_model_tiers():
    """Test budget vs premium model system."""
    print("🧪 Testing Budget vs Premium Model System")
    print("=" * 80)
    
    async with create_server_session() as session:
        
        # Check available tools
        tools = await session.list_tools()
        tool_names = [tool.name for tool in tools.tools]
        print(f"📋 Available tools: {', '.join(tool_names)}")
        
        # Test cases
        test_cases = [
            {
                "name": "Budget Model - FLUX Schnell Default (landscape)",
                "tool": "generate_blog_image",
                "params": {
                    "prompt": "Professional blog header image showing modern office workspace with laptops and coffee",
                    "aspect_ratio": "landscape"
                },
                "expected_cost_range": (0.002, 0.004),
                "expected_model": "black-forest-labs/flux-schnell"
            },
            {
                "name": "Budget Model - FLUX Schnell Explicit (square)",
                "tool": "generate_blog_image", 
                "params": {
                    "prompt": "Square social media image of team collaboration in modern office",
                    "aspect_ratio": "square",
                    "model": "black-forest-labs/flux-schnell"
                },
                "expected_cost_range": (0.002, 0.004),
                "expected_model": "black-forest-labs/flux-schnell"
            },
            {
                "name": "Budget Model - FLUX Schnell (budget premium)",
                "tool": "generate_blog_image",
                "params": {
                    "prompt": "Ultra-wide blog header for technology article with servers and data visualization",
                    "aspect_ratio": "21:9",
                    "model": "black-forest-labs/flux-schnell"
                },
                "expected_cost_range": (0.002, 0.004),
                "expected_model": "black-forest-labs/flux-schnell"
            },
            {
                "name": "Premium Model - FLUX Dev (quality premium)",
                "tool": "generate_premium_image",
                "params": {
                    "prompt": "Professional portrait format image of developer working on complex code",
                    "aspect_ratio": "9:16",
                    "model": "black-forest-labs/flux-dev"
                },
                "expected_cost_range": (0.025, 0.035),
                "expected_model": "black-forest-labs/flux-dev"
            }
        ]
        
        results = {}
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 Test {i}: {test_case['name']}")
            print(f"   Tool: {test_case['tool']}")
            print(f"   Parameters: {test_case['params']}")
            print(f"   Expected cost range: ${test_case['expected_cost_range'][0]:.3f} - ${test_case['expected_cost_range'][1]:.3f}")
            
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
                
                print(f"   ✅ Generation successful!")
                print(f"      Model used: {actual_model}")
                print(f"      Cost: ${actual_cost:.4f}")
                print(f"      Dimensions: {dimensions[0]}x{dimensions[1]}")
                print(f"      Format: {format_type}")
                print(f"      Image path: {response_data.get('image_url', 'N/A')}")
                
                # Validate expectations
                cost_ok = test_case['expected_cost_range'][0] <= actual_cost <= test_case['expected_cost_range'][1]
                model_ok = actual_model == test_case['expected_model']
                
                if cost_ok and model_ok:
                    print(f"      ✅ Cost and model match expectations")
                else:
                    if not cost_ok:
                        print(f"      ⚠️ Cost outside expected range")
                    if not model_ok:
                        print(f"      ⚠️ Model mismatch - expected {test_case['expected_model']}")
                
                # Check aspect ratio worked
                aspect_ratio = test_case['params']['aspect_ratio']
                if aspect_ratio == "landscape" or aspect_ratio == "16:9":
                    aspect_ok = dimensions[0] > dimensions[1]
                    print(f"      {'✅' if aspect_ok else '⚠️'} Landscape aspect ratio: {aspect_ok}")
                elif aspect_ratio == "portrait" or aspect_ratio == "9:16":
                    aspect_ok = dimensions[1] > dimensions[0]
                    print(f"      {'✅' if aspect_ok else '⚠️'} Portrait aspect ratio: {aspect_ok}")
                elif aspect_ratio == "square" or aspect_ratio == "1:1":
                    aspect_ok = abs(dimensions[0] - dimensions[1]) < 50  # Allow small differences
                    print(f"      {'✅' if aspect_ok else '⚠️'} Square aspect ratio: {aspect_ok}")
                elif aspect_ratio == "21:9":
                    aspect_ok = dimensions[0] / dimensions[1] > 2.0  # Ultra-wide
                    print(f"      {'✅' if aspect_ok else '⚠️'} Ultra-wide 21:9 ratio: {aspect_ok}")
                
                results[test_case['name']] = True
                
            except Exception as e:
                print(f"   ❌ Test failed with exception: {e}")
                results[test_case['name']] = False
        
        # Summary
        print(f"\n" + "=" * 80)
        print("📊 BUDGET vs PREMIUM MODEL TEST RESULTS")
        print("=" * 80)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name[:60]:60} {status}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Budget/Premium model system working correctly.")
            print("\n💡 USAGE RECOMMENDATIONS:")
            print("   • Use generate_blog_image() for cost-effective, high-volume content")
            print("   • Use generate_premium_image() for marketing materials, hero images")
            print("   • SDXL: ~400 images per $1 - Perfect for blog content") 
            print("   • FLUX: Higher cost but superior quality and text rendering")
            return True
        else:
            print("❌ Some tests failed. Check the output above.")
            return False

if __name__ == "__main__":
    success = asyncio.run(test_model_tiers())
    exit(0 if success else 1)