#!/usr/bin/env python3
"""
MCP Client Test Script

Tests the MCP Image Generation Server using the official MCP client.
This script will start the server and test all available tools.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def create_server_session():
    """Create an MCP client session connected to our image generation server."""
    
    # Server command
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "mcp_image_server/server.py"],
        env=None
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                logger.info("✅ Connected to MCP image generation server")
                yield session
    except Exception as e:
        logger.error(f"❌ Failed to connect to MCP server: {e}")
        raise


async def test_list_tools(session: ClientSession):
    """Test listing available tools."""
    logger.info("🔍 Testing tool listing...")
    
    try:
        result = await session.list_tools()
        tools = result.tools
        
        logger.info(f"📋 Found {len(tools)} tools:")
        for tool in tools:
            logger.info(f"   • {tool.name}: {tool.description}")
        
        expected_tools = {
            "generate_blog_image",
            "list_generated_images", 
            "get_cache_stats",
            "clear_cache"
        }
        
        tool_names = {tool.name for tool in tools}
        missing_tools = expected_tools - tool_names
        
        if missing_tools:
            logger.error(f"❌ Missing expected tools: {missing_tools}")
            return False
        
        logger.info("✅ All expected tools found")
        return True
        
    except Exception as e:
        logger.error(f"❌ Tool listing failed: {e}")
        return False


async def test_cache_stats(session: ClientSession):
    """Test getting cache statistics."""
    logger.info("📊 Testing cache statistics...")
    
    try:
        result = await session.call_tool("get_cache_stats", {})
        
        if result.isError:
            logger.error(f"❌ Cache stats failed: {result.content}")
            return False
        
        stats = json.loads(result.content[0].text)
        logger.info(f"💾 Cache enabled: {stats.get('cache_enabled', 'unknown')}")
        logger.info(f"📁 Cache directory: {stats.get('cache_directory', 'unknown')}")
        
        logger.info("✅ Cache stats retrieved successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Cache stats test failed: {e}")
        return False


async def test_list_images(session: ClientSession):
    """Test listing generated images."""
    logger.info("🖼️ Testing image listing...")
    
    try:
        result = await session.call_tool("list_generated_images", {})
        
        if result.isError:
            logger.error(f"❌ List images failed: {result.content}")
            return False
        
        data = json.loads(result.content[0].text)
        count = data.get('count', 0)
        storage_path = data.get('storage_path', 'unknown')
        
        logger.info(f"📂 Found {count} existing images")
        logger.info(f"📁 Storage path: {storage_path}")
        
        logger.info("✅ Image listing successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Image listing test failed: {e}")
        return False


async def test_image_generation(session: ClientSession):
    """Test actual image generation."""
    logger.info("🎨 Testing image generation...")
    
    # Check if API token is configured
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token or api_token == "your_replicate_api_token_here":
        logger.warning("⚠️ Skipping image generation test - API token not configured")
        return True
    
    try:
        # Test parameters
        params = {
            "prompt": "A simple test image of a sunset over mountains",
            "style": "photorealistic", 
            "aspect_ratio": "landscape"
        }
        
        logger.info(f"🚀 Generating image with prompt: '{params['prompt']}'")
        logger.info("   This may take 30-60 seconds...")
        
        start_time = time.time()
        result = await session.call_tool("generate_blog_image", params)
        generation_time = time.time() - start_time
        
        if result.isError:
            logger.error(f"❌ Image generation failed: {result.content}")
            return False
        
        data = json.loads(result.content[0].text)
        
        # Check for error in response
        if "error" in data:
            logger.error(f"❌ Generation error: {data['error']}")
            return False
        
        image_url = data.get('image_url', '')
        cost = data.get('cost', 0)
        cached = data.get('cached', False)
        
        logger.info(f"✅ Image generated in {generation_time:.1f}s")
        logger.info(f"💰 Cost: ${cost:.4f}")
        logger.info(f"💾 Cached: {cached}")
        logger.info(f"📁 Image path: {image_url}")
        
        # Verify file exists
        if image_url and Path(image_url).exists():
            file_size = Path(image_url).stat().st_size
            logger.info(f"📄 File size: {file_size:,} bytes")
            logger.info("✅ Image generation test successful")
            return True
        else:
            logger.error(f"❌ Generated image file not found: {image_url}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Image generation test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def test_cache_clear(session: ClientSession):
    """Test clearing cache."""
    logger.info("🧹 Testing cache clearing...")
    
    try:
        result = await session.call_tool("clear_cache", {})
        
        if result.isError:
            logger.error(f"❌ Cache clear failed: {result.content}")
            return False
        
        data = json.loads(result.content[0].text)
        removed = data.get('removed_entries', 0)
        
        logger.info(f"🗑️ Removed {removed} expired cache entries")
        logger.info("✅ Cache clear successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Cache clear test failed: {e}")
        return False


async def main():
    """Run comprehensive MCP server tests."""
    print("🧪 MCP Image Generation Server Test Suite")
    print("=" * 50)
    
    # Verify environment
    env_file = Path(".env")
    if env_file.exists():
        logger.info("✅ .env file found")
    else:
        logger.warning("⚠️ .env file not found - some tests may be skipped")
    
    test_results = {}
    
    try:
        async with create_server_session() as session:
            
            # Run all tests
            tests = [
                ("List Tools", test_list_tools),
                ("Cache Stats", test_cache_stats), 
                ("List Images", test_list_images),
                ("Image Generation", test_image_generation),
                ("Cache Clear", test_cache_clear),
            ]
            
            for test_name, test_func in tests:
                logger.info(f"\n📋 Running test: {test_name}")
                try:
                    result = await test_func(session)
                    test_results[test_name] = result
                    if result:
                        logger.info(f"✅ {test_name} PASSED")
                    else:
                        logger.error(f"❌ {test_name} FAILED")
                except Exception as e:
                    logger.error(f"❌ {test_name} EXCEPTION: {e}")
                    test_results[test_name] = False
    
    except Exception as e:
        logger.error(f"❌ Failed to create server session: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = sum(test_results.values())
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! MCP server is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())