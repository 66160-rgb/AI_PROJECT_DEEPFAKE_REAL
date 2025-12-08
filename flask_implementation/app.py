from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ExifTags
import io, json, os

app = Flask(__name__)

MODEL_PATH = "model/my_model_after_history2.h5"
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

def do_ela(img, q=90):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    comp = Image.open(io.BytesIO(buf.getvalue()))
    diff = ImageChops.difference(img, comp)
    return ImageEnhance.Brightness(diff).enhance(5)

def get_exif(img):
    try:
        raw = img._getexif()
        return {ExifTags.TAGS.get(k, k): v for k, v in raw.items()} if raw else {}
    except:
        return {}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    exif_data = None
    ela_img_path = None
    uploaded_img_path = None

    if request.method == "POST":
        file = request.files["image"]
        img = Image.open(file).convert("RGB")

        arr = np.expand_dims(np.array(img.resize(IMG_SIZE)) / 255.0, 0)
        p = float(model.predict(arr)[0][0])

        label = "Fake" if p >= 0.5 else "Real"
        confidence = round(p * 100, 2)
        prediction = label

        ela = do_ela(img)
        ela_path = "static/ela.jpg"
        img_path = "static/upload.jpg"

        ela.save(ela_path)
        img.save(img_path)

        uploaded_img_path = img_path
        ela_img_path = ela_path

        exif_data = json.dumps(get_exif(img), indent=2)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           exif_data=exif_data,
                           uploaded_img=uploaded_img_path,
                           ela_img=ela_img_path)

if __name__ == "__main__":
    app.run(debug=True)
