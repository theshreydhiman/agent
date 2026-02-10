"""High-level image generation interface wrapping ComfyUI (Flux + IP-Adapter)."""

from __future__ import annotations

import asyncio
import base64
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from config import get_settings
from generation.image.comfyui_client import ComfyUIClient, ComfyUIError
from generation.image.prompts import get_negative_prompt

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    """Container returned after a successful image generation."""

    image_bytes: bytes
    file_path: str
    seed: int
    generation_time_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Workflow builder helpers
# ---------------------------------------------------------------------------

def _encode_image_to_base64(path: str) -> str:
    """Read an image file and return its base64-encoded contents."""
    return base64.b64encode(Path(path).read_bytes()).decode()


# ---------------------------------------------------------------------------
# ImageGenerator
# ---------------------------------------------------------------------------

class ImageGenerator:
    """Generate character images via ComfyUI using Flux + IP-Adapter.

    Usage::

        gen = ImageGenerator()
        result = await gen.generate_image("a woman in a coffee shop")
        print(result.file_path)
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = ComfyUIClient(settings.comfyui_base_url)
        self._output_dir = Path(settings.comfyui_output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # -- workflow construction -------------------------------------------------

    @staticmethod
    def build_workflow(
        prompt: str,
        negative_prompt: str = "",
        width: int = 1080,
        height: int = 1080,
        reference_image_path: str | None = None,
        seed: int | None = None,
    ) -> dict:
        """Build a ComfyUI workflow dict for Flux + IP-Adapter.

        The workflow graph:
            LoadCheckpoint -> CLIPTextEncode (pos) ─┐
                              CLIPTextEncode (neg) ──┤
                              EmptyLatentImage ──────┤
                                                     ├─> KSampler -> VAEDecode -> SaveImage
            (optional) LoadImage -> IPAdapter ────────┘

        When *reference_image_path* is provided an IP-Adapter node is injected
        so that the generated image preserves the reference character's identity.
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        if not negative_prompt:
            negative_prompt = get_negative_prompt()

        settings = get_settings()

        workflow: dict[str, Any] = {
            # -- UNET loader (Flux model) --
            "1": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": "flux1-dev.safetensors",
                    "weight_dtype": "default",
                },
            },
            # -- Dual CLIP loader (T5 + CLIP-L for Flux) --
            "8": {
                "class_type": "DualCLIPLoader",
                "inputs": {
                    "clip_name1": "t5xxl_fp16.safetensors",
                    "clip_name2": "clip_l.safetensors",
                    "type": "flux",
                },
            },
            # -- VAE loader (Flux autoencoder) --
            "9": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": "ae.safetensors",
                },
            },
            # -- CLIP positive prompt --
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["8", 0],
                },
            },
            # -- empty latent --
            "4": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1,
                },
            },
            # -- KSampler (Flux parameters) --
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["2", 0],  # Flux doesn't use negative prompts
                    "latent_image": ["4", 0],
                    "seed": seed,
                    "steps": 20,  # Flux typically uses fewer steps
                    "cfg": 1.0,  # Flux uses very low CFG
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                },
            },
            # -- VAE decode --
            "6": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["9", 0],
                },
            },
            # -- save image --
            "7": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["6", 0],
                    "filename_prefix": "flux_gen",
                },
            },
        }

        # -- optional IP-Adapter for reference-based generation ----------------
        # NOTE: IP-Adapter is currently disabled for Flux (not yet compatible)
        # Re-enable when Flux-compatible IP-Adapter models are available
        if False and reference_image_path is not None:
            image_b64 = _encode_image_to_base64(reference_image_path)

            # Load the reference image
            workflow["10"] = {
                "class_type": "LoadImage",
                "inputs": {
                    "image": image_b64,
                },
            }

            # IP-Adapter model loader
            workflow["11"] = {
                "class_type": "IPAdapterModelLoader",
                "inputs": {
                    "ipadapter_file": "ip-adapter-plus-face_sdxl_vit-h.safetensors",
                },
            }

            # CLIP Vision loader
            workflow["12"] = {
                "class_type": "CLIPVisionLoader",
                "inputs": {
                    "clip_name": "clip-vit-large-patch14.safetensors",
                },
            }

            # CLIP Vision encode
            workflow["13"] = {
                "class_type": "CLIPVisionEncode",
                "inputs": {
                    "clip_vision": ["12", 0],
                    "image": ["10", 0],
                },
            }

            # IP-Adapter apply (patches the model)
            workflow["14"] = {
                "class_type": "IPAdapterApply",
                "inputs": {
                    "ipadapter": ["11", 0],
                    "clip_vision_output": ["13", 0],
                    "image": ["10", 0],
                    "model": ["1", 0],
                    "weight": 0.85,
                    "noise": 0.0,
                },
            }

            # Rewire KSampler to use IP-Adapter-patched model
            workflow["5"]["inputs"]["model"] = ["14", 0]

        return workflow

    # -- single generation -----------------------------------------------------

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1080,
        height: int = 1080,
        reference_image_path: str | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        """Generate a single image and persist it locally.

        Returns a ``GenerationResult`` with the raw bytes, saved path, seed,
        and timing information.
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        workflow = self.build_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            reference_image_path=reference_image_path,
            seed=seed,
        )

        logger.info(
            "image_generation_started",
            width=width,
            height=height,
            seed=seed,
            has_reference=reference_image_path is not None,
        )

        t0 = time.monotonic()

        try:
            prompt_id = await self._client.queue_prompt(workflow)
            await self._client.wait_for_completion(prompt_id)
            images = await self._client.get_output_images(prompt_id)
        except ComfyUIError:
            logger.exception("image_generation_failed", seed=seed)
            raise

        elapsed = round(time.monotonic() - t0, 2)

        if not images:
            raise ComfyUIError("Generation completed but no images returned")

        image_bytes = images[0]
        file_name = f"{uuid.uuid4().hex}.png"
        file_path = self._output_dir / file_name
        file_path.write_bytes(image_bytes)

        logger.info(
            "image_generation_completed",
            file_path=str(file_path),
            seed=seed,
            elapsed=elapsed,
            size_bytes=len(image_bytes),
        )

        return GenerationResult(
            image_bytes=image_bytes,
            file_path=str(file_path),
            seed=seed,
            generation_time_seconds=elapsed,
            metadata={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "reference_image_path": reference_image_path,
            },
        )

    # -- batch generation ------------------------------------------------------

    async def generate_batch(
        self,
        prompts: list[str],
        width: int = 1080,
        height: int = 1080,
        reference_image_path: str | None = None,
    ) -> list[GenerationResult]:
        """Generate multiple images concurrently.

        Each prompt receives its own random seed.  Results are returned in the
        same order as *prompts*.
        """
        logger.info("batch_generation_started", count=len(prompts))

        tasks = [
            self.generate_image(
                prompt=p,
                width=width,
                height=height,
                reference_image_path=reference_image_path,
            )
            for p in prompts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successes from failures and log any errors.
        final: list[GenerationResult] = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "batch_item_failed",
                    index=idx,
                    error=str(result),
                )
                raise result
            final.append(result)

        logger.info("batch_generation_completed", count=len(final))
        return final

    # -- cleanup ---------------------------------------------------------------

    async def close(self) -> None:
        """Release underlying HTTP resources."""
        await self._client.close()
