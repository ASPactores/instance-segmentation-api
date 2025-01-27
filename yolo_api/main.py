from fastapi import FastAPI, APIRouter, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import base64
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

app = FastAPI(
    title="YOLOv9 API",
    root_path="/api/yolo-inference",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Limit this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoculeCount(BaseModel):
    photo: bytes
    count: int

@app.get("/")
def read_root():
    return {"Hello": "From YOLOv9 API!"}

@app.post("/predict/", response_model=LoculeCount)
async def count_locules(input: UploadFile) -> LoculeCount:
    model = YOLO("yolo_model.pt")
    locule_count = 0
    try:
        image = await input.read()
        photo_stream = io.BytesIO(image)
        image = Image.open(photo_stream)
        image_array = np.array(image)

        if image_array.ndim == 2:  # Grayscale to RGB
            image_array = np.stack((image_array,) * 3, axis=-1)
        elif image_array.shape[2] == 4:  # RGBA to RGB
            image_array = image_array[:, :, :3]

        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        results = model(image_array, conf=0.8)
        objects = results[0].boxes
        detected_durian = objects.cls.tolist()
        locule_count = len(detected_durian)

        image_with_masks = results[0].plot()
        success, encoded_image = cv2.imencode(".jpg", image_with_masks)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to encode image",
            )

        return {
            "count": locule_count,
            "photo": base64.b64encode(encoded_image).decode(),
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing image: {e}",
        )

