"""
OpenAI Images API provider (direct, billed to OPENAI_API_KEY).

Models that Replicate does not host (gpt-image-2.5) are called here. Generation
without references uses /images/generations; with references, /images/edits.
"""

import asyncio
import base64
import logging
import mimetypes
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import requests

from mcp_image_server.providers.base import BaseProvider

logger = logging.getLogger(__name__)

OPENAI_API_BASE = "https://api.openai.com/v1"
REQUEST_TIMEOUT_S = 600
REFERENCE_DOWNLOAD_TIMEOUT_S = 60
REFERENCE_MAX_BYTES = 50 * 1024 * 1024  # OpenAI's per-image limit for edits
ERROR_BODY_MAX_CHARS = 500

DEFAULT_SIZE = "1024x1024"
DEFAULT_QUALITY = "medium"
DEFAULT_OUTPUT_FORMAT = "png"
DEFAULT_REFERENCE_NAME = "reference.png"
DEFAULT_REFERENCE_MIME = "image/png"

# 1024x1024 prices from OpenAI's token estimator (2026-09); larger sizes cost more.
GPT_IMAGE_25_COST_PER_QUALITY = {
    "low": 0.006,
    "medium": 0.013,
    "high": 0.053,
    "auto": 0.053,  # model picks; priced as high
}


def _gpt_image_25(friendly_name: str, api_model: str) -> Dict[str, Any]:
    return {
        "friendly_name": friendly_name,
        "api_model": api_model,
        "cost_per_run": GPT_IMAGE_25_COST_PER_QUALITY[DEFAULT_QUALITY],
        "cost_per_quality": GPT_IMAGE_25_COST_PER_QUALITY,
        "max_width": 3840,
        "max_height": 3840,
        "aspect_ratio_mode": "openai_size",
        "category": "premium",
        "supports_quality": list(GPT_IMAGE_25_COST_PER_QUALITY),
        "default_quality": DEFAULT_QUALITY,
        "image_input_param": "image",
        "image_input_is_list": True,
    }


class OpenAIImagesProvider(BaseProvider):
    MODEL_CONFIGS = {
        "openai/gpt-image-2.5-flare": _gpt_image_25("GPT-Image-2.5 Flare (OpenAI)", "gpt-image-2.5-flare"),
        "openai/gpt-image-2.5-sunburst": _gpt_image_25("GPT-Image-2.5 Sunburst (OpenAI)", "gpt-image-2.5-sunburst"),
    }

    async def _generate_image(self, prompt: str, model: str, ctx: Any, **kwargs) -> Dict[str, Any]:
        config = self.MODEL_CONFIGS[model]
        options = {
            "size": kwargs.get("size", DEFAULT_SIZE),
            "quality": kwargs.get("quality") or config["default_quality"],
            "output_format": kwargs.get("output_format") or DEFAULT_OUTPUT_FORMAT,
        }
        self.log_generation_details(prompt, model, **options)
        fields = {"model": config["api_model"], "prompt": prompt, **options}
        image_bytes = await asyncio.to_thread(self._request, fields, kwargs.get("image"))

        suffix = ".jpg" if options["output_format"] == "jpeg" else f".{options['output_format']}"
        with tempfile.NamedTemporaryFile(prefix="openai_image_", suffix=suffix, delete=False) as out:
            out.write(image_bytes)
        logger.info(f"OpenAI generation completed - saved to temp file: {out.name}")
        return {"image_url": out.name, "model_used": config["api_model"]}

    def _request(self, fields: Dict[str, str], references: Optional[List[Union[str, Path]]]) -> bytes:
        if references:
            endpoint = "edits"
            body = {"data": fields, "files": [("image[]", _reference_file(ref)) for ref in references]}
        else:
            endpoint = "generations"
            body = {"json": fields}
        response = requests.post(f"{OPENAI_API_BASE}/images/{endpoint}",
                                 headers={"Authorization": f"Bearer {self.api_token}"},
                                 timeout=REQUEST_TIMEOUT_S, **body)

        if not response.ok:
            try:
                error = response.json().get("error") or {}
            except ValueError:
                error = {"message": response.text}
            # The code (e.g. insufficient_quota) is kept in the message so
            # is_non_retryable can recognise billing/auth failures.
            raise Exception(
                f"OpenAI API error {response.status_code} ({error.get('code') or error.get('type')}): "
                f"{str(error.get('message'))[:ERROR_BODY_MAX_CHARS]}"
            )
        return base64.b64decode(response.json()["data"][0]["b64_json"])

    def estimate_cost(self, model: str, quality: str = None) -> float:
        config = self.MODEL_CONFIGS[model]
        return config["cost_per_quality"].get(quality, config["cost_per_run"])

    def get_model_friendly_name(self, model: str) -> str:
        return self.MODEL_CONFIGS[model]["friendly_name"]


def _reference_file(ref: Union[str, Path]) -> Tuple[str, bytes, str]:
    """A multipart (name, bytes, mime) tuple for a local path or an http(s) URL."""
    if isinstance(ref, Path):
        name, data = ref.name, ref.read_bytes()
    else:
        with requests.get(ref, timeout=REFERENCE_DOWNLOAD_TIMEOUT_S, stream=True) as response:
            response.raise_for_status()
            data = response.raw.read(REFERENCE_MAX_BYTES + 1, decode_content=True)
        name = Path(urlparse(ref).path).name or DEFAULT_REFERENCE_NAME
    if len(data) > REFERENCE_MAX_BYTES:
        raise ValueError(f"Reference image {name} exceeds {REFERENCE_MAX_BYTES // (1024 * 1024)} MB")
    return name, data, mimetypes.guess_type(name)[0] or DEFAULT_REFERENCE_MIME
