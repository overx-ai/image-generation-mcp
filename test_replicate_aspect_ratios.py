#!/usr/bin/env python3
"""
Test Replicate aspect ratio parameters with real generation.
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

async def test_replicate_aspect_ratios():
    """Test various aspect ratio formats."""
    print("🧪 Testing Replicate Aspect Ratio Parameters")
    print("=" * 70)
    
    # Test cases with different aspect ratio formats
    test_cases = [
        {
            "name": "User-friendly landscape (16:9)",
            "params": {
                "prompt": "A wide landscape view of mountains and lakes for testing aspect ratios - landscape format",
                "aspect_ratio": "landscape"
            },
            "expected_replicate_ratio": "16:9"
        },
        {
            "name": "User-friendly square (1:1)", 
            "params": {
                "prompt": "A square format image of a modern office workspace for testing - square format",
                "aspect_ratio": "square"
            },
            "expected_replicate_ratio": "1:1"
        },
        {
            "name": "Direct Replicate ratio (4:3)",
            "params": {
                "prompt": "A classic 4:3 format photo of a business meeting for testing aspect ratios",
                "aspect_ratio": "4:3"
            },
            "expected_replicate_ratio": "4:3"
        },
        {
            "name": "Direct Replicate ratio (9:16 portrait)",
            "params": {
                "prompt": "A tall portrait format image of a person working on mobile device",
                "aspect_ratio": "9:16"
            },
            "expected_replicate_ratio": "9:16"
        },
        {
            "name": "Blog header format (21:9 ultrawide)",
            "params": {
                "prompt": "An ultrawide blog header image showing a technology conference",
                "aspect_ratio": "21:9"
            },
            "expected_replicate_ratio": "21:9"
        }
    ]
    
    async with create_server_session() as session:
        results = {}
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 Test {i}: {test_case['name']}")
            print(f"   Aspect ratio parameter: {test_case['params']['aspect_ratio']}")
            print(f"   Expected Replicate format: {test_case['expected_replicate_ratio']}")
            
            try:
                result = await session.call_tool("generate_blog_image", test_case["params"])
                
                if result.isError:
                    print(f"   ❌ Tool call failed: {result.content}")
                    results[test_case['name']] = False
                    continue
                
                response_data = json.loads(result.content[0].text)
                
                if "error" in response_data:
                    print(f"   ❌ Generation failed: {response_data['error']}")
                    results[test_case['name']] = False
                    continue
                
                # Check results
                print(f"   ✅ Generation successful!")
                print(f"      Image path: {response_data.get('image_url', 'N/A')}")
                print(f"      User aspect ratio: {response_data.get('aspect_ratio', 'N/A')}")
                print(f"      Dimensions: {response_data.get('dimensions', 'N/A')}")
                print(f"      Format: {response_data.get('format', 'N/A')}")
                print(f"      Cost: ${response_data.get('cost', 0):.4f}")
                
                # Check metadata file
                image_path = Path(response_data.get('image_url', ''))
                metadata_path = image_path.with_suffix('.json')
                
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    
                    requested_ratio = metadata.get('requested_aspect_ratio', 'N/A')
                    actual_dims = metadata.get('actual_dimensions', {})
                    
                    print(f"      Metadata - Requested ratio: {requested_ratio}")
                    print(f"      Metadata - Actual dims: {actual_dims.get('width', 0)}x{actual_dims.get('height', 0)}")
                    
                    # Verify the aspect ratio was properly mapped
                    if requested_ratio == test_case['expected_replicate_ratio']:
                        print(f"      ✅ Aspect ratio properly mapped to Replicate format")
                    else:
                        print(f"      ⚠️ Aspect ratio mapping issue - expected {test_case['expected_replicate_ratio']}, got {requested_ratio}")
                
                results[test_case['name']] = True
                
            except Exception as e:
                print(f"   ❌ Test failed with exception: {e}")
                results[test_case['name']] = False
        
        # Summary
        print(f"\n" + "=" * 70)
        print("📊 ASPECT RATIO TEST RESULTS")
        print("=" * 70)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name[:50]:50} {status}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL ASPECT RATIO TESTS PASSED! Replicate integration working correctly.")
            return True
        else:
            print("❌ Some tests failed. Check the output above.")
            return False

if __name__ == "__main__":
    success = asyncio.run(test_replicate_aspect_ratios())
    exit(0 if success else 1)