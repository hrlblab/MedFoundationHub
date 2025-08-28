# utils.py
from PIL import Image

def preprocess_image(image_input):
    image = image_input if isinstance(image_input, Image.Image) else Image.open(image_input)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Do NOT force resize here; VLM processor will handle it.
    return image
