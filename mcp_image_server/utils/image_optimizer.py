"""
Image optimization utilities for web deployment.

This module provides functions to optimize images for web use by:
- Resizing images to appropriate dimensions
- Converting formats for better web compatibility
- Applying compression without significant quality loss
- Creating responsive image variants
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from PIL import Image
import io

logger = logging.getLogger(__name__)


class ImageOptimizer:
    """Handles image optimization for web deployment."""
    
    # Maximum dimensions for web images (can be overridden)
    DEFAULT_MAX_WIDTH = 2048
    DEFAULT_MAX_HEIGHT = 2048
    
    # Quality settings for different formats
    JPEG_QUALITY = 85  # Good balance between quality and file size
    WEBP_QUALITY = 85
    PNG_COMPRESS_LEVEL = 6  # 0-9, where 9 is maximum compression
    
    # Responsive image breakpoints (width in pixels)
    RESPONSIVE_BREAKPOINTS = [
        320,   # Mobile small
        640,   # Mobile large
        1024,  # Tablet
        1920,  # Desktop
    ]
    
    @staticmethod
    def optimize_for_web(
        input_path: Path,
        output_path: Optional[Path] = None,
        max_width: Optional[int] = None,
        max_height: Optional[int] = None,
        format: Optional[str] = None,
        quality: Optional[int] = None,
        create_webp: bool = True,
        create_responsive: bool = False
    ) -> Dict[str, Any]:
        """
        Optimize an image for web deployment.
        
        Args:
            input_path: Path to the input image
            output_path: Path for the optimized image (if None, replaces original)
            max_width: Maximum width (default: 2048)
            max_height: Maximum height (default: 2048)
            format: Target format ('jpeg', 'png', 'webp', None for auto)
            quality: Quality setting (1-100, None for format defaults)
            create_webp: Also create a WebP version
            create_responsive: Create responsive image variants
            
        Returns:
            Dict containing optimization results and metrics
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        if output_path is None:
            output_path = input_path
        
        max_width = max_width or ImageOptimizer.DEFAULT_MAX_WIDTH
        max_height = max_height or ImageOptimizer.DEFAULT_MAX_HEIGHT
        
        logger.info(f"Optimizing image: {input_path}")
        
        try:
            # Open and get original image info
            with Image.open(input_path) as img:
                original_size = input_path.stat().st_size
                original_dimensions = img.size
                original_format = img.format
                
                # Convert RGBA to RGB if necessary for JPEG
                if format == 'jpeg' or (format is None and original_format == 'JPEG'):
                    if img.mode in ('RGBA', 'LA', 'P'):
                        # Create white background
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                        img = background
                
                # Resize if needed
                img_optimized = ImageOptimizer._resize_image(img, max_width, max_height)
                
                # Determine output format
                if format:
                    output_format = format.upper()
                elif original_format in ['JPEG', 'PNG', 'WEBP']:
                    output_format = original_format
                else:
                    # Default to JPEG for photos, PNG for graphics/transparency
                    output_format = 'PNG' if img.mode in ('RGBA', 'LA', 'P') else 'JPEG'
                
                # Set quality/compression parameters
                save_kwargs = ImageOptimizer._get_save_params(output_format, quality)
                
                # Save optimized image
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Ensure correct file extension
                correct_ext = ImageOptimizer._get_extension(output_format)
                if output_path.suffix.lower() != correct_ext:
                    output_path = output_path.with_suffix(correct_ext)
                
                img_optimized.save(output_path, format=output_format, **save_kwargs)
                
                optimized_size = output_path.stat().st_size
                
                result = {
                    'success': True,
                    'output_path': str(output_path),
                    'original_size': original_size,
                    'optimized_size': optimized_size,
                    'size_reduction': round((1 - optimized_size / original_size) * 100, 2),
                    'original_dimensions': original_dimensions,
                    'optimized_dimensions': img_optimized.size,
                    'format': output_format,
                    'variants': []
                }
                
                # Create WebP version if requested
                if create_webp and output_format != 'WEBP':
                    webp_path = output_path.with_suffix('.webp')
                    webp_kwargs = ImageOptimizer._get_save_params('WEBP', quality)
                    img_optimized.save(webp_path, format='WEBP', **webp_kwargs)
                    webp_size = webp_path.stat().st_size
                    
                    result['webp_variant'] = {
                        'path': str(webp_path),
                        'size': webp_size,
                        'size_reduction': round((1 - webp_size / original_size) * 100, 2)
                    }
                    logger.info(f"Created WebP variant: {webp_path} ({webp_size:,} bytes)")
                
                # Create responsive variants if requested
                if create_responsive:
                    variants = ImageOptimizer._create_responsive_variants(
                        img, output_path, output_format, quality
                    )
                    result['variants'] = variants
                
                logger.info(
                    f"Optimization complete: {original_size:,} -> {optimized_size:,} bytes "
                    f"({result['size_reduction']}% reduction)"
                )
                
                return result
                
        except Exception as e:
            logger.error(f"Error optimizing image: {e}")
            return {
                'success': False,
                'error': str(e),
                'output_path': str(output_path)
            }
    
    @staticmethod
    def _resize_image(img: Image.Image, max_width: int, max_height: int) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        width, height = img.size
        
        # Check if resizing is needed
        if width <= max_width and height <= max_height:
            return img
        
        # Calculate scaling factor
        scale = min(max_width / width, max_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Use high-quality resampling
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    @staticmethod
    def _get_save_params(format: str, quality: Optional[int] = None) -> Dict[str, Any]:
        """Get save parameters for different formats."""
        params = {'optimize': True}
        
        if format == 'JPEG':
            params['quality'] = quality or ImageOptimizer.JPEG_QUALITY
            params['progressive'] = True
        elif format == 'PNG':
            params['compress_level'] = ImageOptimizer.PNG_COMPRESS_LEVEL
        elif format == 'WEBP':
            params['quality'] = quality or ImageOptimizer.WEBP_QUALITY
            params['method'] = 6  # Slowest/best compression
        
        return params
    
    @staticmethod
    def _get_extension(format: str) -> str:
        """Get file extension for format."""
        extensions = {
            'JPEG': '.jpg',
            'PNG': '.png',
            'WEBP': '.webp'
        }
        return extensions.get(format, '.jpg')
    
    @staticmethod
    def _create_responsive_variants(
        img: Image.Image,
        base_path: Path,
        format: str,
        quality: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Create responsive image variants for different screen sizes."""
        variants = []
        width, height = img.size
        
        for breakpoint in ImageOptimizer.RESPONSIVE_BREAKPOINTS:
            # Skip if image is already smaller than breakpoint
            if width <= breakpoint:
                continue
            
            # Calculate new dimensions
            scale = breakpoint / width
            new_height = int(height * scale)
            
            # Resize image
            img_resized = img.resize((breakpoint, new_height), Image.Resampling.LANCZOS)
            
            # Save variant
            variant_path = base_path.parent / f"{base_path.stem}_{breakpoint}w{base_path.suffix}"
            save_kwargs = ImageOptimizer._get_save_params(format, quality)
            img_resized.save(variant_path, format=format, **save_kwargs)
            
            variants.append({
                'width': breakpoint,
                'path': str(variant_path),
                'size': variant_path.stat().st_size
            })
            
            logger.info(f"Created responsive variant: {breakpoint}w")
        
        return variants
    
    @staticmethod
    def batch_optimize(
        input_dir: Path,
        output_dir: Optional[Path] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Batch optimize all images in a directory.
        
        Args:
            input_dir: Directory containing images to optimize
            output_dir: Output directory (if None, replaces originals)
            **kwargs: Additional arguments passed to optimize_for_web
            
        Returns:
            List of optimization results
        """
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
        
        for image_path in input_dir.rglob('*'):
            if image_path.suffix.lower() in image_extensions:
                if output_dir:
                    relative_path = image_path.relative_to(input_dir)
                    output_path = output_dir / relative_path
                else:
                    output_path = None
                
                result = ImageOptimizer.optimize_for_web(
                    image_path, output_path, **kwargs
                )
                result['input_path'] = str(image_path)
                results.append(result)
        
        logger.info(f"Batch optimization complete: {len(results)} images processed")
        return results
    
    @staticmethod
    def estimate_optimization(input_path: Path) -> Dict[str, Any]:
        """
        Estimate optimization potential without actually optimizing.
        
        Args:
            input_path: Path to the image
            
        Returns:
            Dict containing estimated optimization metrics
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        with Image.open(input_path) as img:
            width, height = img.size
            format = img.format
            current_size = input_path.stat().st_size
            
            # Estimate sizes for different formats
            estimates = {}
            
            # Estimate JPEG size (typically 10-20% of uncompressed)
            if format != 'JPEG':
                estimates['jpeg'] = {
                    'estimated_size': current_size * 0.3,
                    'estimated_reduction': 70
                }
            
            # Estimate WebP size (typically 25-35% smaller than JPEG)
            if format != 'WEBP':
                jpeg_size = estimates.get('jpeg', {}).get('estimated_size', current_size * 0.5)
                estimates['webp'] = {
                    'estimated_size': jpeg_size * 0.7,
                    'estimated_reduction': round((1 - jpeg_size * 0.7 / current_size) * 100, 2)
                }
            
            # Check if resizing would help
            max_width = ImageOptimizer.DEFAULT_MAX_WIDTH
            max_height = ImageOptimizer.DEFAULT_MAX_HEIGHT
            
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_pixels = (width * scale) * (height * scale)
                original_pixels = width * height
                size_ratio = new_pixels / original_pixels
                
                estimates['resize'] = {
                    'new_dimensions': (int(width * scale), int(height * scale)),
                    'estimated_size': current_size * size_ratio,
                    'estimated_reduction': round((1 - size_ratio) * 100, 2)
                }
            
            return {
                'current_size': current_size,
                'current_dimensions': (width, height),
                'current_format': format,
                'optimization_potential': estimates
            }