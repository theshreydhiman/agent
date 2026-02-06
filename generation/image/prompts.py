"""Prompt-building utilities for Flux + IP-Adapter image generation."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Instagram resolution presets
# ---------------------------------------------------------------------------

INSTAGRAM_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "feed_square": (1080, 1080),
    "feed_portrait": (1080, 1350),
    "story": (1080, 1920),
    "reel": (1080, 1920),
}

# ---------------------------------------------------------------------------
# Scene presets
# ---------------------------------------------------------------------------

SCENE_PRESETS: dict[str, str] = {
    "urban_street": (
        "standing on a modern city sidewalk, urban architecture, graffiti walls, "
        "golden hour sunlight, street photography vibe"
    ),
    "coffee_shop": (
        "sitting inside a trendy artisan coffee shop, warm ambient lighting, "
        "latte art on the table, exposed brick walls, cozy atmosphere"
    ),
    "beach_sunset": (
        "on a sandy beach at golden hour, warm sunset glow, gentle waves, "
        "soft ocean breeze, tropical paradise"
    ),
    "gym": (
        "in a modern well-equipped gym, professional lighting, workout equipment "
        "in background, energetic fitness atmosphere"
    ),
    "home_cozy": (
        "in a stylish cozy living room, soft natural window light, plush blankets, "
        "candles, warm earthy tones, homey aesthetic"
    ),
    "office_modern": (
        "in a sleek modern office with floor-to-ceiling windows, minimalist decor, "
        "natural daylight, professional setting"
    ),
    "garden": (
        "in a lush blooming garden, surrounded by flowers and greenery, dappled "
        "sunlight through trees, serene nature setting"
    ),
    "rooftop": (
        "on a trendy rooftop terrace at dusk, city skyline panorama in background, "
        "string lights, lounge furniture, evening ambiance"
    ),
    "library": (
        "in an elegant library with tall wooden bookshelves, warm reading lamp, "
        "vintage leather armchair, intellectual aesthetic"
    ),
    "art_gallery": (
        "inside a contemporary art gallery, white walls, dramatic spotlighting, "
        "abstract artwork in background, sophisticated atmosphere"
    ),
    "park": (
        "in a sunlit city park, autumn foliage, tree-lined path, soft natural "
        "lighting, relaxed outdoor setting"
    ),
    "restaurant": (
        "seated at an upscale restaurant, elegant table setting, soft candlelight, "
        "fine dining ambiance, bokeh background"
    ),
    "studio_portrait": (
        "in a professional photography studio, clean backdrop, studio lighting "
        "with softbox, high-end editorial look"
    ),
    "travel_landmark": (
        "standing before an iconic travel landmark, tourist destination, vivid "
        "blue sky, vibrant colors, travel photography style"
    ),
    "night_city": (
        "on a neon-lit city street at night, reflections on wet pavement, "
        "cinematic moody lighting, cyberpunk urban atmosphere"
    ),
}

# ---------------------------------------------------------------------------
# Outfit presets
# ---------------------------------------------------------------------------

OUTFIT_PRESETS: dict[str, str] = {
    "casual_chic": (
        "wearing a casual chic outfit, fitted jeans, tucked-in blouse, "
        "minimal gold jewelry, clean white sneakers"
    ),
    "athleisure": (
        "wearing stylish athleisure, matching sports bra and high-waisted "
        "leggings, running shoes, fitness watch"
    ),
    "formal_elegant": (
        "wearing an elegant formal gown, floor-length dress, statement earrings, "
        "heels, polished glamorous look"
    ),
    "streetwear": (
        "wearing trendy streetwear, oversized graphic hoodie, cargo pants, "
        "chunky sneakers, bucket hat, layered accessories"
    ),
    "summer_dress": (
        "wearing a flowy floral summer dress, sandals, sun hat, light fabric, "
        "breezy warm-weather style"
    ),
    "winter_cozy": (
        "wearing a cozy winter outfit, oversized knit sweater, wool scarf, "
        "beanie, warm-toned palette, layered textures"
    ),
    "business_casual": (
        "wearing sharp business casual, tailored blazer, slacks, loafers, "
        "understated watch, polished professional look"
    ),
    "evening_glam": (
        "wearing glamorous evening attire, sequin mini dress, stiletto heels, "
        "bold statement necklace, smokey eye makeup"
    ),
    "bohemian": (
        "wearing bohemian style, flowing maxi skirt, crochet top, layered "
        "bracelets, ankle boots, free-spirited aesthetic"
    ),
    "minimalist": (
        "wearing a minimalist monochrome outfit, clean lines, neutral tones, "
        "simple silhouette, understated elegance"
    ),
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_character_prompt(
    character_config: dict,
    scene: str = "",
    outfit: str = "",
    style: str = "",
) -> str:
    """Assemble a coherent Stable-Diffusion / Flux prompt from character config.

    Parameters
    ----------
    character_config:
        Dictionary describing the character.  Expected keys (all optional):
        ``name``, ``gender``, ``age``, ``ethnicity``, ``hair``, ``eyes``,
        ``body_type``, ``distinguishing_features``, ``base_prompt``.
    scene:
        Either a key from ``SCENE_PRESETS`` or a free-form scene description.
    outfit:
        Either a key from ``OUTFIT_PRESETS`` or a free-form outfit description.
    style:
        Additional stylistic directions (e.g. "cinematic", "editorial").

    Returns
    -------
    str
        A single comma-separated prompt string.
    """
    parts: list[str] = []

    # Base prompt override (full description provided by the user).
    base = character_config.get("base_prompt", "")
    if base:
        parts.append(base)
    else:
        # Build description from individual attributes.
        gender = character_config.get("gender", "woman")
        age = character_config.get("age", "")
        ethnicity = character_config.get("ethnicity", "")
        hair = character_config.get("hair", "")
        eyes = character_config.get("eyes", "")
        body_type = character_config.get("body_type", "")
        features = character_config.get("distinguishing_features", "")

        subject = " ".join(
            filter(None, [ethnicity, f"{age}-year-old" if age else "", gender])
        )
        parts.append(f"photo of a {subject}".strip())

        if hair:
            parts.append(hair)
        if eyes:
            parts.append(eyes)
        if body_type:
            parts.append(body_type)
        if features:
            parts.append(features)

    # Scene -------------------------------------------------------------------
    if scene:
        scene_text = SCENE_PRESETS.get(scene, scene)
        parts.append(scene_text)

    # Outfit ------------------------------------------------------------------
    if outfit:
        outfit_text = OUTFIT_PRESETS.get(outfit, outfit)
        parts.append(outfit_text)

    # Style -------------------------------------------------------------------
    if style:
        parts.append(style)

    # Quality boosters (always appended) --------------------------------------
    parts.append(
        "photorealistic, ultra detailed, 8k uhd, DSLR quality, "
        "natural skin texture, professional photography"
    )

    return ", ".join(parts)


def get_negative_prompt() -> str:
    """Return a standard negative prompt optimised for photorealistic outputs."""
    return (
        "cartoon, anime, illustration, painting, drawing, sketch, "
        "3d render, cgi, unrealistic, deformed, disfigured, bad anatomy, "
        "bad proportions, extra limbs, mutated hands, fused fingers, "
        "too many fingers, long neck, blurry, out of focus, low quality, "
        "low resolution, watermark, text, logo, signature, cropped, "
        "oversaturated, overexposed, underexposed, duplicate, "
        "ugly, distorted face, plastic skin"
    )
