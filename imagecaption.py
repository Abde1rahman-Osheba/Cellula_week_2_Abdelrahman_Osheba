from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

_processor = None
_model = None


def _load_model():
    """Load BLIP processor and model once, cache globally."""
    global _processor, _model
    if _processor is None or _model is None:
        _processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        _model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return _processor, _model


def generate_caption(image: Image.Image, prompt: str = "a photography of") -> str:
    """Generate a text caption for the given PIL Image using BLIP-1."""
    processor, model = _load_model()
    raw_image = image.convert("RGB")
    inputs = processor(raw_image, prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption
