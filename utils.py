import numpy as np
from PIL import Image

ALLOWED_EXTENSIONS = {"png","jpg","jpeg"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(path):
    img = Image.open(path).resize((224,224))
    return np.array(img)
