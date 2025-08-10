"""
Replicate API provider for image generation.

Implements the Replicate API integration with model mapping,
friendly names, and cost estimation.
"""

import asyncio
import logging
from typing import Any, Dict

import replicate
from .base import BaseProvider

logger = logging.getLogger(__name__)


class ReplicateProvider(BaseProvider):
    """
    Replicate API provider implementation.
    
    Features:
    - Model mapping with friendly names
    - Cost estimation per model
    - Async API integration
    - Comprehensive error handling
    """
    
    # Model configurations with costs and metadata
    MODEL_CONFIGS = {
        # Budget Models - Custom aspect ratios via width/height
        "stability-ai/sdxl": {
            "friendly_name": "SDXL (Budget)",
            "cost_per_run": 0.0025,  # ~400 images per $1 
            "max_width": 1024,
            "max_height": 1024,
            "aspect_ratio_mode": "width_height",  # Uses width/height parameters
            "category": "budget"
        },
        "bytedance/sdxl-lightning-4step": {
            "friendly_name": "SDXL-Lightning (Budget)",
            "cost_per_run": 0.002,  # ~500 images per $1 
            "max_width": 1024,
            "max_height": 1024,
            "aspect_ratio_mode": "width_height",  # Uses width/height parameters
            "category": "budget"
        },
        "stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f36bc82acaa43cc671ba16622c0b040067a0098": {
            "friendly_name": "Stable Diffusion v1.5",
            "cost_per_run": 0.018,
            "max_width": 1024,
            "max_height": 1024,
            "aspect_ratio_mode": "width_height",
            "category": "budget"
        },
        
        # Budget Models - Native aspect ratio support
        "black-forest-labs/flux-schnell": {
            "friendly_name": "FLUX.1 [schnell] (Budget Premium)",
            "cost_per_run": 0.003,  # ~333 images per $1
            "max_width": 1440,
            "max_height": 1440,
            "aspect_ratio_mode": "native",  # Uses aspect_ratio parameter
            "category": "budget"
        },
        
        # Premium Models - Native aspect ratio support
        "black-forest-labs/flux-dev": {
            "friendly_name": "FLUX.1 [dev] (Quality Premium)",
            "cost_per_run": 0.030,  # ~33 images per $1
            "max_width": 1440,
            "max_height": 1440,
            "aspect_ratio_mode": "native",
            "category": "premium"
        },
        "black-forest-labs/flux-1.1-pro": {
            "friendly_name": "FLUX 1.1 [pro] (Ultra Premium)",
            "cost_per_run": 0.040,  # ~25 images per $1
            "max_width": 1440,
            "max_height": 1440,
            "aspect_ratio_mode": "native",
            "category": "premium"
        },
        "recraft-ai/recraft-v3": {
            "friendly_name": "Recraft V3 (Design Premium)",
            "cost_per_run": 0.040,  # ~25 images per $1
            "max_width": 1440,
            "max_height": 1440,
            "aspect_ratio_mode": "native",
            "category": "premium"
        },
        "bytedance/seedream-3": {
            "friendly_name": "Seedream-3 (Ultra Premium)",
            "cost_per_run": 0.050,  # ~20 images per $1
            "max_width": 2048,
            "max_height": 2048,
            "aspect_ratio_mode": "native",
            "category": "premium"
        },
        "ideogram-ai/ideogram-v3-turbo": {
            "friendly_name": "Ideogram V3 Turbo (Text Specialist)",
            "cost_per_run": 0.035,  # ~28 images per $1, excellent for text in images
            "max_width": 2048,
            "max_height": 2048,
            "aspect_ratio_mode": "native",
            "category": "premium"
        },
        
        # Aliases for backward compatibility
        "flux-schnell": {
            "friendly_name": "FLUX.1 [schnell]",
            "cost_per_run": 0.003,
            "max_width": 1440,
            "max_height": 1440,
            "actual_model": "black-forest-labs/flux-schnell",
            "aspect_ratio_mode": "native",
            "category": "premium"
        },
        "flux-dev": {
            "friendly_name": "FLUX.1 [dev]", 
            "cost_per_run": 0.030,
            "max_width": 1440,
            "max_height": 1440,
            "actual_model": "black-forest-labs/flux-dev",
            "aspect_ratio_mode": "native",
            "category": "premium"
        }
    }
    
    def __init__(self, api_token: str, config: Dict[str, Any]):
        """
        Initialize Replicate provider.
        
        Args:
            api_token: Replicate API token
            config: Configuration dictionary
        """
        super().__init__(api_token, config)
        
        # Initialize Replicate client
        self.client = replicate.Client(api_token=api_token)
        logger.info("Replicate provider initialized successfully")
    
    async def _generate_image(self, prompt: str, model: str, ctx: Any, **kwargs) -> Dict[str, Any]:
        """
        Generate image using Replicate API.
        
        Args:
            prompt: Text description for image generation
            model: Replicate model identifier
            ctx: MCP context for logging
            **kwargs: Generation parameters (width, height, etc.)
            
        Returns:
            Dictionary with image_url and metadata
        """
        # Resolve model alias to actual model ID
        actual_model = self._resolve_model(model)
        
        # Prepare input parameters
        model_input = {"prompt": prompt}
        model_input.update(kwargs)
        
        # Log generation details
        log_params = {k: v for k, v in model_input.items() if k != 'prompt'}
        self.log_generation_details(prompt, actual_model, **log_params)
        
        try:
            # Run generation using asyncio.to_thread for async compatibility
            output = await asyncio.to_thread(
                self.client.run,
                actual_model,
                input=model_input
            )
            
            # Extract image URL from output
            if isinstance(output, list) and len(output) > 0:
                first_output = output[0]
                if hasattr(first_output, 'url'):
                    image_url = first_output.url
                else:
                    image_url = str(first_output)
            elif hasattr(output, 'url'):
                image_url = output.url
            else:
                image_url = str(output)
            
            logger.info(f"Replicate generation completed - URL: {image_url}")
            
            return {
                "image_url": image_url,
                "provider_response": output,
                "model_used": actual_model
            }
            
        except replicate.exceptions.ReplicateError as e:
            logger.error(f"Replicate API error: {e}")
            raise Exception(f"Replicate API error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error in Replicate generation: {e}")
            raise
    
    def _resolve_model(self, model: str) -> str:
        """
        Resolve model alias to actual Replicate model ID.
        
        Args:
            model: Model identifier (can be alias or full ID)
            
        Returns:
            Actual Replicate model identifier
        """
        if model in self.MODEL_CONFIGS:
            config = self.MODEL_CONFIGS[model]
            actual_model = config.get("actual_model", model)
            logger.debug(f"Resolved model '{model}' to '{actual_model}'")
            return actual_model
        
        # If not in config, assume it's a direct model ID
        logger.debug(f"Using model ID directly: '{model}'")
        return model
    
    def estimate_cost(self, model: str) -> float:
        """
        Estimate cost for generating an image with the given model.
        
        Args:
            model: Model identifier
            
        Returns:
            Estimated cost in USD
        """
        resolved_model = self._resolve_model(model)
        
        if resolved_model in self.MODEL_CONFIGS:
            cost = self.MODEL_CONFIGS[resolved_model]["cost_per_run"]
            logger.debug(f"Cost estimate for {model}: ${cost:.4f}")
            return cost
        elif model in self.MODEL_CONFIGS:
            cost = self.MODEL_CONFIGS[model]["cost_per_run"]
            logger.debug(f"Cost estimate for {model}: ${cost:.4f}")
            return cost
        
        # Default cost estimate for unknown models
        default_cost = 0.025
        logger.warning(f"Unknown model {model}, using default cost estimate: ${default_cost:.4f}")
        return default_cost
    
    def get_model_friendly_name(self, model: str) -> str:
        """
        Get human-friendly name for the model.
        
        Args:
            model: Model identifier
            
        Returns:
            Human-readable model name
        """
        resolved_model = self._resolve_model(model)
        
        if resolved_model in self.MODEL_CONFIGS:
            name = self.MODEL_CONFIGS[resolved_model]["friendly_name"]
            logger.debug(f"Friendly name for {model}: {name}")
            return name
        elif model in self.MODEL_CONFIGS:
            name = self.MODEL_CONFIGS[model]["friendly_name"]
            logger.debug(f"Friendly name for {model}: {name}")
            return name
        
        # Return the model ID if no friendly name found
        logger.debug(f"No friendly name for {model}, returning model ID")
        return model
    
    def validate_model(self, model: str) -> bool:
        """
        Validate if a model is supported.
        
        Args:
            model: Model identifier
            
        Returns:
            True if model is supported
        """
        resolved_model = self._resolve_model(model)
        is_valid = resolved_model in self.MODEL_CONFIGS or model in self.MODEL_CONFIGS
        logger.debug(f"Model validation for {model}: {is_valid}")
        return is_valid
    
    def get_supported_models(self) -> list[str]:
        """
        Get list of supported models.
        
        Returns:
            List of supported model identifiers
        """
        models = list(self.MODEL_CONFIGS.keys())
        logger.debug(f"Supported models: {models}")
        return models
    
    def get_model_limits(self, model: str) -> Dict[str, int]:
        """
        Get dimensional limits for a model.
        
        Args:
            model: Model identifier
            
        Returns:
            Dictionary with max_width and max_height
        """
        resolved_model = self._resolve_model(model)
        
        if resolved_model in self.MODEL_CONFIGS:
            config = self.MODEL_CONFIGS[resolved_model]
            limits = {
                "max_width": config.get("max_width", 1024),
                "max_height": config.get("max_height", 1024)
            }
            logger.debug(f"Model limits for {model}: {limits}")
            return limits
        elif model in self.MODEL_CONFIGS:
            config = self.MODEL_CONFIGS[model]
            limits = {
                "max_width": config.get("max_width", 1024),
                "max_height": config.get("max_height", 1024)
            }
            logger.debug(f"Model limits for {model}: {limits}")
            return limits
        
        # Default limits for unknown models
        default_limits = {"max_width": 1024, "max_height": 1024}
        logger.warning(f"Unknown model {model}, using default limits: {default_limits}")
        return default_limits