import os, json
from PIL import Image


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_image(img, path):
    img.save(path)


def save_text(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
