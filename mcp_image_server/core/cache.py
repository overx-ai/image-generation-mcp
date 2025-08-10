"""
Image caching system for the MCP Image Server.

This module provides a file-based cache system with TTL expiration and size management.
Cache keys are generated using SHA256 hashing of prompt, model, and parameters.
"""

import json
import logging
import hashlib
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ImageCache:
    """
    File-based image cache with TTL and size management.
    
    Features:
    - SHA256-based cache keys
    - TTL-based expiration (30 days default)
    - LRU cleanup when size limits exceeded
    - Atomic operations with metadata stored in JSON sidecar files
    """
    
    def __init__(self, cache_dir: str = "./cache", ttl_days: int = 30, max_size_mb: int = 1000):
        """
        Initialize the image cache.
        
        Args:
            cache_dir: Directory to store cached images
            ttl_days: Time to live in days for cached items
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.ttl_days = ttl_days
        self.max_size_mb = max_size_mb
        self.ttl_seconds = ttl_days * 24 * 3600
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache initialized - Directory: {cache_dir}, TTL: {ttl_days} days, Max size: {max_size_mb} MB")
    
    def generate_cache_key(self, prompt: str, model: str, style: str = "photorealistic", 
                          aspect_ratio: str = "landscape", width: int = 1024, height: int = 768) -> str:
        """
        Generate a SHA256 cache key from generation parameters.
        
        Args:
            prompt: Text prompt for image generation
            model: Model identifier
            style: Image style
            aspect_ratio: Aspect ratio setting
            width: Image width
            height: Image height
            
        Returns:
            SHA256 hash as hex string
        """
        # Normalize inputs for consistent hashing
        normalized_prompt = prompt.strip().lower()
        cache_input = f"{normalized_prompt}|{model}|{style}|{aspect_ratio}|{width}x{height}"
        
        cache_key = hashlib.sha256(cache_input.encode('utf-8')).hexdigest()
        logger.debug(f"Generated cache key: {cache_key} for input: {cache_input}")
        return cache_key
    
    def _get_cache_paths(self, cache_key: str) -> tuple[Path, Path]:
        """Get file paths for cache entry and its metadata."""
        cache_file = self.cache_dir / f"{cache_key}.png"
        metadata_file = self.cache_dir / f"{cache_key}.json"
        return cache_file, metadata_file
    
    def check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Check if a cache entry exists and is valid.
        
        Args:
            cache_key: SHA256 cache key
            
        Returns:
            Cache entry data if valid, None if not found or expired
        """
        cache_file, metadata_file = self._get_cache_paths(cache_key)
        
        if not cache_file.exists() or not metadata_file.exists():
            logger.debug(f"Cache miss: {cache_key} - files not found")
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check if cache entry has expired
            cached_time = metadata.get('cached_at', 0)
            current_time = time.time()
            
            if current_time - cached_time > self.ttl_seconds:
                logger.info(f"Cache expired: {cache_key} - removing expired entry")
                self._remove_cache_entry(cache_key)
                return None
            
            # Return cache hit with file path
            metadata['image_path'] = str(cache_file.absolute())
            logger.info(f"Cache hit: {cache_key}")
            return metadata
            
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error reading cache metadata for {cache_key}: {e}")
            self._remove_cache_entry(cache_key)
            return None
    
    def save_to_cache(self, cache_key: str, image_data: bytes, metadata: Dict[str, Any]) -> bool:
        """
        Save an image and its metadata to cache.
        
        Args:
            cache_key: SHA256 cache key
            image_data: Binary image data
            metadata: Metadata dictionary
            
        Returns:
            True if successfully saved, False otherwise
        """
        try:
            cache_file, metadata_file = self._get_cache_paths(cache_key)
            
            # Add cache timestamp
            metadata['cached_at'] = time.time()
            metadata['cache_key'] = cache_key
            
            # Write image data
            with open(cache_file, 'wb') as f:
                f.write(image_data)
            
            # Write metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved to cache: {cache_key} ({len(image_data)} bytes)")
            
            # Check and enforce cache size limits
            self._enforce_size_limit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving to cache {cache_key}: {e}")
            return False
    
    def _remove_cache_entry(self, cache_key: str) -> None:
        """Remove a cache entry and its metadata."""
        try:
            cache_file, metadata_file = self._get_cache_paths(cache_key)
            
            if cache_file.exists():
                cache_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()
                
            logger.debug(f"Removed cache entry: {cache_key}")
        except Exception as e:
            logger.warning(f"Error removing cache entry {cache_key}: {e}")
    
    def clear_expired(self) -> int:
        """
        Clear expired cache entries.
        
        Returns:
            Number of entries removed
        """
        removed_count = 0
        current_time = time.time()
        
        try:
            for metadata_file in self.cache_dir.glob("*.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    cached_time = metadata.get('cached_at', 0)
                    if current_time - cached_time > self.ttl_seconds:
                        cache_key = metadata_file.stem
                        self._remove_cache_entry(cache_key)
                        removed_count += 1
                        
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Error reading metadata file {metadata_file}: {e}")
                    # Remove corrupted metadata file
                    metadata_file.unlink(missing_ok=True)
                    cache_file = metadata_file.with_suffix('.png')
                    cache_file.unlink(missing_ok=True)
                    removed_count += 1
                    
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")
        
        logger.info(f"Cache cleanup completed - removed {removed_count} expired entries")
        return removed_count
    
    def _enforce_size_limit(self) -> None:
        """Enforce cache size limit using LRU eviction."""
        total_size = self._get_cache_size_mb()
        
        if total_size <= self.max_size_mb:
            return
        
        logger.info(f"Cache size {total_size:.1f}MB exceeds limit {self.max_size_mb}MB - starting cleanup")
        
        # Get all cache entries sorted by access time (oldest first)
        cache_entries = []
        for metadata_file in self.cache_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                cached_time = metadata.get('cached_at', 0)
                cache_key = metadata_file.stem
                cache_file = metadata_file.with_suffix('.png')
                
                if cache_file.exists():
                    file_size = cache_file.stat().st_size
                    cache_entries.append((cached_time, cache_key, file_size))
                    
            except Exception as e:
                logger.warning(f"Error reading cache entry {metadata_file}: {e}")
        
        # Sort by timestamp (oldest first) and remove until under limit
        cache_entries.sort()
        removed_count = 0
        
        for cached_time, cache_key, file_size in cache_entries:
            self._remove_cache_entry(cache_key)
            removed_count += 1
            total_size -= file_size / (1024 * 1024)  # Convert to MB
            
            if total_size <= self.max_size_mb * 0.8:  # Leave 20% headroom
                break
        
        logger.info(f"LRU cleanup completed - removed {removed_count} entries, new size: {total_size:.1f}MB")
    
    def _get_cache_size_mb(self) -> float:
        """Get total cache size in MB."""
        total_size = 0
        try:
            for file_path in self.cache_dir.glob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.error(f"Error calculating cache size: {e}")
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics and information.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            image_files = list(self.cache_dir.glob("*.png"))
            metadata_files = list(self.cache_dir.glob("*.json"))
            
            total_size_mb = self._get_cache_size_mb()
            
            stats = {
                "cache_directory": str(self.cache_dir.absolute()),
                "total_entries": len(image_files),
                "metadata_files": len(metadata_files),
                "total_size_mb": round(total_size_mb, 2),
                "max_size_mb": self.max_size_mb,
                "ttl_days": self.ttl_days,
                "usage_percent": round((total_size_mb / self.max_size_mb) * 100, 1) if self.max_size_mb > 0 else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "error": str(e),
                "cache_directory": str(self.cache_dir.absolute()),
                "total_entries": 0,
                "total_size_mb": 0
            }