---
status: current
created: 2026-07-22
updated: 2026-10-04
---

# 002 - Model Fallback & Per-Model Inputs

**Prerequisites**: [001 - Image Generation Tools](001-image-generation-tools.md)
**Related**: [000 - Changelog](000-changelog.md)

How the server survives a model failing, and how each model's Replicate input is
built. Motivation: `openai/gpt-image-2` runs on Replicate but bills the OpenAI
account behind the forwarded `OPENAI_API_KEY`. When that account is dry, every
generation died with `ModelError: Your token does not have sufficient credits.
(E001)` even though the Replicate token was fine.

## The chain

`generate_image()` builds an ordered attempt list and walks it until one succeeds.
Each rung gets the provider's own retry logic (`max_attempts`, backoff + jitter).

| # | Attempt | Applies when |
|---|---------|--------------|
| 1 | requested model (default `openai/gpt-image-2.5-flare`, OpenAI-direct), own `OPENAI_API_KEY` | always, except an OpenAI-direct model with no key set |
| 2 | **same model without our OpenAI key** → Replicate-billed | `OPENAI_API_KEY` set **and** model is `aspect_ratio_mode: gpt_image` |
| 3 | `models.fallback[0]` — `openai/gpt-image-2`, Replicate-billed | candidate ≠ requested model |
| 4 | `models.fallback[1]` — `nano-banana` | candidate ≠ requested model |
| 5 | `models.fallback[2]` — `bytedance/seedream-3` | candidate ≠ requested model |

OpenAI-direct models (served by `OpenAIImagesProvider`, see `is_openai_direct`)
are dropped from the chain when `OPENAI_API_KEY` is unset, so the default still
works on a Replicate-only setup.

Rung 2 alone recovers from the dry/invalid-OpenAI-key case — the same model runs,
billed by Replicate, with identical output quality.

`nano-banana` is a **family**, resolved to a concrete model by the requested quality
so a fallback is never a quality downgrade:

| `quality` | Fallback model | Resolution |
|-----------|----------------|------------|
| `low`, `medium` | `google/nano-banana-2` | 2K |
| `high`, `auto` | `google/nano-banana-pro` | 2K |

When `reference_image` was passed, candidates without an `image_input_param` are
dropped (seedream-3 goes; gpt-image-2 and both nano-bananas stay). If that empties
the list, the first attempt is kept so the caller still gets the explanatory
`ValueError` naming the models that do support references.

**Result contract**: `model` and `cost` always describe what actually ran;
`fallback_from` names the requested model when a substitute produced the image
(`None` otherwise). Same fields in the JSON metadata sidecar.

**Cache**: keys are computed by the tools from the *requested* model, before
generation. A fallback image is **not** cached: stored under the requested
model's key it would be served for the whole TTL, so one transient OpenAI
failure would pin that prompt to the substitute model for 30 days. A repeat
request retries the requested model instead (billing/auth errors fail fast,
see `is_non_retryable`).

**File**: `mcp_image_server/server.py` — `build_attempts`, `resolve_fallback_model`,
`NANO_BANANA_BY_QUALITY`, and the attempt loop in `generate_image`.

```
build_attempts(config, model, quality, reference_images)
        │  [(model, use_openai_key), …]
        ▼
   for candidate in attempts:
        build_model_input(candidate, …, use_openai_key)
        provider.generate(...)  ──success──▶ selected_model = candidate; break
              │ failure (after provider retries)
              └─▶ next candidate; last one re-raises
```

## Configuring the chain

```yaml
# mcp_image_server/config/config.yaml
models:
  fallback:
    - "nano-banana"              # family alias, resolved by quality
    - "bytedance/seedream-3"
```

Entries are model ids, or the `nano-banana` alias. Empty/absent list = no model
switching (rung 2 still applies).

## Non-retryable errors

Retrying a billing or auth failure cannot help; it only delays the next rung by the
full backoff schedule. `providers/base.py` exposes `is_non_retryable(error)` and
`generate()` breaks out of the retry loop on a match.

| Matched substring (case-insensitive) | Source |
|---|---|
| `sufficient credits`, `insufficient`, `(e001)` | out of credit |
| `invalid api key`, `api key is invalid`, `incorrect api key`, `(e002)` | bad key |
| `authentication`, `unauthorized` | auth |

## Per-model inputs

Replicate models disagree on their input schemas, so `build_model_input()` is driven
entirely by `ReplicateProvider.MODEL_CONFIGS` (`providers/replicate.py`):

| Key | Effect |
|-----|--------|
| `aspect_ratio_mode` | `gpt_image` (restricted ratio enum + `quality`), `native` (`aspect_ratio` input), `width_height` (SDXL) |
| `supports_megapixels` | adds `megapixels: "1"` — **FLUX only**; recraft/seedream/ideogram/nano-banana reject it |
| `supported_ratios` | allowed `aspect_ratio` enum; anything else is clamped (below) |
| `extra_input` | fixed extra inputs, e.g. `{"resolution": "2K"}` for the nano-bananas (model default is 1K) |
| `supports_output_format` | sends `output_format` (default `png`) — nano-banana defaults to jpg otherwise |
| `image_input_param` / `image_input_is_list` | where reference images go (`input_images` for gpt-image-2, `image_input` for nano-banana) |
| `cost_per_run` / `cost_per_quality` | cost reporting only, never billing |

### Aspect-ratio clamping

`clamp_aspect_ratio(ratio, supported)` maps a ratio onto the numerically nearest
value the model accepts, so a fallback never fails on a ratio the primary model
allowed. Examples against seedream-3 (`4:5`, `5:4`, `9:21` unsupported):

| Requested | seedream-3 gets |
|-----------|-----------------|
| `16:9` | `16:9` (passthrough) |
| `4:5` | `3:4` |
| `9:21` | `9:16` |

gpt-image-2 keeps its own dedicated map (`GPT_IMAGE_ASPECT_MAP`) because its enum
mixes ratios and fixed pixel sizes.

### Adding a model

1. Add a `MODEL_CONFIGS` entry — read the live schema first:
   `curl -H "Authorization: Bearer $REPLICATE_API_TOKEN" https://api.replicate.com/v1/models/<owner>/<name>`
   and set `supported_ratios` / `supports_megapixels` / `supports_output_format`
   from its actual `Input` properties.
2. Optionally list it under `models.premium` in `config.yaml` (selectable) and/or
   `models.fallback` (automatic).
3. Extend `_demo_fallback()` in `server.py` if it introduces a new input rule.

## Verification

```bash
uv run python -m mcp_image_server.server --demo     # chain + input building, no API

# force the model-switch rungs against the real API
uv run python -c "
import asyncio
from mcp_image_server.providers.replicate import ReplicateProvider
from mcp_image_server.server import generate_blog_image
orig = ReplicateProvider._generate_image
async def patched(self, prompt, model, ctx, **kw):
    if model.startswith('openai/'):
        raise Exception('ModelError: Your token does not have sufficient credits. (E001)')
    return await orig(self, prompt, model, ctx, **kw)
ReplicateProvider._generate_image = patched
print(asyncio.run(generate_blog_image.fn(prompt='a teal skateboard', aspect_ratio='16:9',
      filename='fb.png', output_dir='test_images'))['fallback_from'])
"
```

Expect `openai/gpt-image-2` as `fallback_from` and `model` = `google/nano-banana-2`
(or `-pro` with `quality='high'`).
