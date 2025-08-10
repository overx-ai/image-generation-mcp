"""
Eden AI provider for image generation.

Note: This is a placeholder implementation. Eden AI integration
is not fully implemented yet.
"""

import logging
from typing import Any, Dict

from .base import BaseProvider

logger = logging.getLogger(__name__)


class EdenAIProvider(BaseProvider):
    """
    Eden AI provider implementation (placeholder).
    
    This provider is not fully implemented yet and will raise
    NotImplementedError for most operations.
    """
    
    def __init__(self, api_token: str, config: Dict[str, Any]):
        """Initialize Eden AI provider."""
        super().__init__(api_token, config)
        logger.warning("Eden AI provider is not fully implemented yet")
    
    async def _generate_image(self, prompt: str, model: str, ctx: Any, **kwargs) -> Dict[str, Any]:
        """Generate image using Eden AI (not implemented)."""
        raise NotImplementedError("Eden AI provider is not implemented yet")
    
    def estimate_cost(self, model: str) -> float:
        """Estimate cost for Eden AI model."""
        # Default estimate for Eden AI
        return 0.020
    
    def get_model_friendly_name(self, model: str) -> str:
        """Get friendly name for Eden AI model."""
        model_names = {
            "stabilityai__stable-diffusion-v1-5": "Stable Diffusion v1.5 (Eden AI)"
        }
        return model_names.get(model, model)
    
    def get_supported_models(self) -> list[str]:
        """Get supported Eden AI models."""
        return [
            "stabilityai__stable-diffusion-v1-5"
        ]