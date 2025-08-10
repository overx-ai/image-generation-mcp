"""
Utility functions and classes for the MCP Image Server.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ImageGenerationError(Exception):
    """
    Custom exception for image generation failures.
    
    This exception is raised when image generation fails after all retry attempts
    have been exhausted.
    """
    
    def __init__(self, message: str, original_exception: Optional[Exception] = None, attempts: int = 0):
        """
        Initialize the ImageGenerationError.
        
        Args:
            message: Error message describing the failure
            original_exception: The original exception that caused the failure
            attempts: Number of attempts made before failing
        """
        self.original_exception = original_exception
        self.attempts = attempts
        
        # Build comprehensive error message
        error_msg = f"Image generation failed after {attempts} attempts: {message}"
        if original_exception:
            error_msg += f" (Original error: {type(original_exception).__name__}: {str(original_exception)})"
        
        super().__init__(error_msg)
        logger.error(f"ImageGenerationError raised: {error_msg}")
    
    def __repr__(self) -> str:
        return (f"ImageGenerationError(message='{str(self)}', "
                f"original_exception={self.original_exception}, "
                f"attempts={self.attempts})")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename by removing or replacing invalid characters."""
    invalid_chars = '<>:"/\\|?*'
    sanitized = filename
    
    for char in invalid_chars:
        sanitized = sanitized.replace(char, '_')
    
    sanitized = sanitized.strip('. ')
    
    if not sanitized:
        sanitized = "untitled"
    
    return sanitized


def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to a maximum length with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def calculate_exponential_backoff(attempt: int, initial_delay: float = 1.0, 
                                backoff_factor: float = 2.0, max_delay: float = 60.0) -> float:
    """Calculate exponential backoff delay with a maximum limit."""
    delay = initial_delay * (backoff_factor ** attempt)
    return min(delay, max_delay)