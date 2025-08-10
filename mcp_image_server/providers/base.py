"""
Base provider class for image generation services.

Provides common functionality including retry logic, cost estimation,
and error handling for all provider implementations.
"""

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from core.utils import calculate_exponential_backoff, format_duration

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """
    Abstract base class for image generation providers.
    
    Handles:
    - Exponential backoff retry with jitter
    - Cost estimation and tracking
    - Common error handling
    - Standardized API interface
    """
    
    def __init__(self, api_token: str, config: Dict[str, Any]):
        """
        Initialize the provider.
        
        Args:
            api_token: API authentication token
            config: Provider configuration dictionary
        """
        self.api_token = api_token
        self.config = config
        self.retry_config = config.get("retry", {})
        
        # Retry settings
        self.max_attempts = self.retry_config.get("max_attempts", 3)
        self.initial_delay = self.retry_config.get("initial_delay", 1.0)
        self.backoff_factor = self.retry_config.get("backoff_factor", 2.0)
        self.max_delay = self.retry_config.get("max_delay", 60.0)
        
        logger.info(f"Initialized {self.__class__.__name__} with retry config: "
                   f"max_attempts={self.max_attempts}, initial_delay={self.initial_delay}s")
    
    @abstractmethod
    async def _generate_image(self, prompt: str, model: str, ctx: Any, **kwargs) -> Dict[str, Any]:
        """
        Provider-specific image generation implementation.
        
        Args:
            prompt: Text description for image generation
            model: Model identifier
            ctx: MCP context for logging
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generation results including image_url
        """
        pass
    
    @abstractmethod
    def estimate_cost(self, model: str) -> float:
        """
        Estimate the cost for generating an image with the given model.
        
        Args:
            model: Model identifier
            
        Returns:
            Estimated cost in USD
        """
        pass
    
    @abstractmethod
    def get_model_friendly_name(self, model: str) -> str:
        """
        Get a human-friendly name for the model.
        
        Args:
            model: Model identifier
            
        Returns:
            Human-readable model name
        """
        pass
    
    async def generate(self, prompt: str, model: str, ctx: Any, **kwargs) -> Dict[str, Any]:
        """
        Generate an image with retry logic and comprehensive error handling.
        
        Args:
            prompt: Text description for image generation
            model: Model identifier  
            ctx: MCP context for logging and user feedback
            **kwargs: Additional generation parameters (width, height, etc.)
            
        Returns:
            Dictionary containing:
            - image_url: URL of generated image
            - generation_time: Time taken in seconds
            - attempts: Number of attempts made
            - estimated_cost: Estimated cost in USD
            - model: Model used for generation
            
        Raises:
            Exception: If all retry attempts fail
        """
        start_time = time.time()
        last_exception = None
        
        logger.info(f"Starting image generation - Model: {model}, Prompt: '{prompt[:50]}...'")
        
        for attempt in range(self.max_attempts):
            try:
                attempt_start = time.time()
                attempt_msg = f"Generation attempt {attempt + 1}/{self.max_attempts}"
                logger.info(attempt_msg)
                
                # Safely handle context info logging
                try:
                    if ctx and hasattr(ctx, 'info') and callable(ctx.info):
                        ctx.info(attempt_msg)
                except Exception as ctx_error:
                    logger.warning(f"Context info logging failed: {ctx_error}")
                
                # Call provider-specific implementation
                result = await self._generate_image(prompt, model, ctx, **kwargs)
                
                # Calculate timing and add metadata
                generation_time = time.time() - start_time
                attempt_time = time.time() - attempt_start
                
                result.update({
                    "generation_time": generation_time,
                    "attempt_time": attempt_time,
                    "attempts": attempt + 1,
                    "estimated_cost": self.estimate_cost(model),
                    "model": model,
                    "provider": self.__class__.__name__
                })
                
                success_msg = (f"Image generation successful on attempt {attempt + 1} "
                             f"in {format_duration(generation_time)}")
                logger.info(success_msg)
                
                # Safely handle context info logging
                try:
                    if ctx and hasattr(ctx, 'info') and callable(ctx.info):
                        ctx.info(success_msg)
                except Exception as ctx_error:
                    logger.warning(f"Context info logging failed: {ctx_error}")
                
                return result
                
            except Exception as e:
                last_exception = e
                attempt_time = time.time() - attempt_start
                
                error_msg = (f"Generation attempt {attempt + 1}/{self.max_attempts} failed "
                           f"after {format_duration(attempt_time)}: {str(e)} ({type(e).__name__})")
                logger.error(error_msg)
                
                # Safely handle context error logging
                try:
                    if ctx and hasattr(ctx, 'error') and callable(ctx.error):
                        ctx.error(error_msg)
                except Exception as ctx_error:
                    logger.warning(f"Context error logging failed: {ctx_error}")
                
                # Don't sleep after the last attempt
                if attempt < self.max_attempts - 1:
                    # Calculate exponential backoff with jitter
                    base_delay = calculate_exponential_backoff(
                        attempt, self.initial_delay, self.backoff_factor, self.max_delay
                    )
                    jitter = random.uniform(0, 0.5)
                    total_delay = base_delay + jitter
                    
                    retry_msg = f"Retrying in {format_duration(total_delay)} (base: {format_duration(base_delay)}, jitter: {format_duration(jitter)})"
                    logger.info(retry_msg)
                    
                    # Safely handle context info logging
                    try:
                        if ctx and hasattr(ctx, 'info') and callable(ctx.info):
                            ctx.info(retry_msg)
                    except Exception as ctx_error:
                        logger.warning(f"Context info logging failed: {ctx_error}")
                    
                    await asyncio.sleep(total_delay)
        
        # All attempts failed
        total_time = time.time() - start_time
        final_error_msg = (f"All {self.max_attempts} generation attempts failed "
                          f"after {format_duration(total_time)}. Last error: {str(last_exception)}")
        logger.error(final_error_msg)
        
        # Safely handle context error logging
        try:
            if ctx and hasattr(ctx, 'error') and callable(ctx.error):
                ctx.error(final_error_msg)
        except Exception as ctx_error:
            logger.warning(f"Context error logging failed: {ctx_error}")
        
        raise last_exception
    
    def log_generation_details(self, prompt: str, model: str, **kwargs) -> None:
        """Log detailed generation parameters for debugging."""
        logger.info(f"Generation details:")
        logger.info(f"  Provider: {self.__class__.__name__}")
        logger.info(f"  Model: {model} ({self.get_model_friendly_name(model)})")
        logger.info(f"  Prompt: {prompt}")
        logger.info(f"  Estimated cost: ${self.estimate_cost(model):.4f}")
        
        for key, value in kwargs.items():
            logger.info(f"  {key}: {value}")
    
    def validate_model(self, model: str) -> bool:
        """
        Validate if a model is supported by this provider.
        
        Args:
            model: Model identifier to validate
            
        Returns:
            True if model is supported, False otherwise
        """
        # Default implementation - providers should override if needed
        return True
    
    def get_supported_models(self) -> list[str]:
        """
        Get list of models supported by this provider.
        
        Returns:
            List of supported model identifiers
        """
        # Default implementation - providers should override
        return []