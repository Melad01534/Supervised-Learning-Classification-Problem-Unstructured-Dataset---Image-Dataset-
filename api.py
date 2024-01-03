from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import io

dim = 100  # dimension of the images
app = FastAPI()

# Load the model
model = load_model(r"C:\Users\20122\Downloads\Cats Vs Dogs Classification\model\my_model.h5")

# CORS setup
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    npimg = np.fromstring(data, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (dim, dim))
    img = img.astype('float32') / 255.0
    img = img.reshape(-1, dim, dim, 1)

    prediction = model.predict(img)
    label = 'dog' if prediction[0][0] > 0.5 else 'cat'
    return {'prediction': label}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
