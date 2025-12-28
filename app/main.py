import tensorflow as tf
import numpy as np
from fastapi import FastAPI,UploadFile, File
from io import BytesIO
from PIL import Image
from fastapi.middleware.cors import  CORSMiddleware
import uvicorn
import os

IMAGE_SIZE = 128

CLASS_NAMES = [
    "Tomato_healthy",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Target_Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Leaf_Mold",
    "Tomato_Late_blight",
    "Tomato_Early_blight",
    "Tomato_Bacterial_spot"
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "saved_model", "1.keras")

MODEL = tf.keras.models.load_model('../saved_model/1.keras')

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image) / 255.0
    return image


@app.get("/ping")
async def ping():
    return {"status": "alive"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)  # (1, H, W, 3)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {
        "class": predicted_class,
        "confidence": round(confidence * 100, 2)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)