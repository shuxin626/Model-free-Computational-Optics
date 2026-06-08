"""Generate the README task illustrations with the OpenAI Images API.

Usage:
    OPENAI_API_KEY=... python utils/generate_readme_aigc_task_images.py

The script writes:
    assets/optical_computing_aigc.png
    assets/cgh_aigc.png
"""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "assets"

PROMPTS = {
    "optical_computing_aigc.png": (
        "Create a clean scientific AIGC illustration for a GitHub README task card: "
        "model-free optical computing. Wide 16:9 composition. A coherent laser beam "
        "passes through a phase object, two spatial light modulator planes, and a camera "
        "sensor; small classification logits appear as abstract bars at the output. Use "
        "a polished academic visualization style, dark neutral lab background with cyan "
        "and magenta optical paths, subtle diffraction patterns, no readable text, no "
        "logos, no watermark. Make it visually clear but not crowded."
    ),
    "cgh_aigc.png": (
        "Create a clean scientific AIGC illustration for a GitHub README task card: "
        "model-free computer-generated holography. Wide 16:9 composition. A phase-only "
        "spatial light modulator displays a colorful optimized phase mask; a laser "
        "reconstructs a target image on a sensor plane through diffraction, with visible "
        "wavefront rings and speckle-like details. Polished academic visualization style, "
        "bright optical table feel, blue-green-gold color accents, no readable text, no "
        "logos, no watermark. Make it distinct from an optical computing classifier "
        "illustration."
    ),
}


def generate_image(client, model, prompt, output_path):
    result = client.images.generate(
        model=model,
        prompt=prompt,
    )
    image_base64 = result.data[0].b64_json
    if image_base64 is None:
        raise RuntimeError("Image generation response did not include b64_json.")
    output_path.write_bytes(base64.b64decode(image_base64))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="gpt-image-2",
        help="OpenAI image model. The current GPT Image API docs use gpt-image-2.",
    )
    args = parser.parse_args()

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    for filename, prompt in PROMPTS.items():
        output_path = ASSET_DIR / filename
        generate_image(client, args.model, prompt, output_path)
        print("wrote {}".format(output_path))


if __name__ == "__main__":
    main()
