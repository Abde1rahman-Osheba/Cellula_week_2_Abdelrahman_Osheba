"""
imagecaption.py
---------------
Image captioning module using BLIP-1 (Salesforce/blip-image-captioning-large).
Provides a reusable generate_caption() function that accepts a PIL Image
and returns a descriptive caption string.
"""

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# ---------------------------------------------------------------------------
# Global model references (lazy-loaded on first call)
# ---------------------------------------------------------------------------
_processor = None
_model = None


def _load_model():
    """Load the BLIP processor and model (called once, cached globally)."""
    global _processor, _model
    if _processor is None or _model is None:
        _processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        _model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
    return _processor, _model


def generate_caption(image: Image.Image, prompt: str = "a photography of") -> str:
    """
    Generate a text caption for the given PIL Image.

    Parameters
    ----------
    image : PIL.Image.Image
        The input image (will be converted to RGB internally).
    prompt : str, optional
        An optional text prompt to condition the caption generation.

    Returns
    -------
    str
        The generated caption.
    """
    processor, model = _load_model()
    raw_image = image.convert("RGB")
    inputs = processor(raw_image, prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption
