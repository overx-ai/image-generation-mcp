#!/usr/bin/env python3
"""
Test mandatory aspect ratio parameter validation.
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

async def test_aspect_ratio_validation():
    """Test aspect ratio parameter validation."""
    print("🧪 Testing Aspect Ratio Parameter Validation")
    print("=" * 60)
    
    async with create_server_session() as session:
        
        # Test cases
        test_cases = [
            {
                "name": "Missing aspect_ratio (should fail)",
                "params": {"prompt": "A test image"},
                "should_fail": True,
                "expected_error": "aspect_ratio parameter is required"
            },
            {
                "name": "Invalid aspect_ratio (should fail)", 
                "params": {"prompt": "A test image", "aspect_ratio": "invalid"},
                "should_fail": True,
                "expected_error": "Invalid aspect_ratio"
            },
            {
                "name": "Valid landscape aspect_ratio",
                "params": {"prompt": "A test landscape image", "aspect_ratio": "landscape"},
                "should_fail": False
            },
            {
                "name": "Valid portrait aspect_ratio",
                "params": {"prompt": "A test portrait image", "aspect_ratio": "portrait"}, 
                "should_fail": False
            },
            {
                "name": "Valid square aspect_ratio",
                "params": {"prompt": "A test square image", "aspect_ratio": "square"},
                "should_fail": False
            }
        ]
        
        results = {}
        
        for test_case in test_cases:
            print(f"\n📋 Testing: {test_case['name']}")
            print(f"   Parameters: {test_case['params']}")
            
            try:
                result = await session.call_tool("generate_blog_image", test_case["params"])
                
                if result.isError:
                    print(f"   ❌ Tool call failed: {result.content}")
                    results[test_case['name']] = False
                    continue
                
                # Parse the response
                response_data = json.loads(result.content[0].text)
                
                # Check if this test should fail
                if test_case["should_fail"]:
                    if "error" in response_data:
                        if test_case["expected_error"] in response_data["error"]:
                            print(f"   ✅ Correctly rejected with expected error")
                            print(f"      Error: {response_data['error']}")
                            results[test_case['name']] = True
                        else:
                            print(f"   ⚠️ Rejected but with unexpected error: {response_data['error']}")
                            results[test_case['name']] = False
                    else:
                        print(f"   ❌ Should have failed but succeeded")
                        results[test_case['name']] = False
                else:
                    # Test should succeed
                    if "error" in response_data:
                        print(f"   ❌ Should have succeeded but failed: {response_data['error']}")
                        results[test_case['name']] = False
                    else:
                        print(f"   ✅ Successfully accepted valid aspect ratio")
                        if "image_url" in response_data:
                            print(f"      Generated image path: {response_data.get('image_url', 'N/A')}")
                        print(f"      Aspect ratio: {response_data.get('aspect_ratio', 'N/A')}")
                        results[test_case['name']] = True
                        
            except Exception as e:
                print(f"   ❌ Test failed with exception: {e}")
                results[test_case['name']] = False
        
        # Summary
        print(f"\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name[:45]:45} {status}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 ALL TESTS PASSED! Aspect ratio validation is working correctly.")
            return True
        else:
            print("❌ Some tests failed. Check the output above.")
            return False

if __name__ == "__main__":
    success = asyncio.run(test_aspect_ratio_validation())
    exit(0 if success else 1)