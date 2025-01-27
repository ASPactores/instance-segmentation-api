import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import traceback

import io
import base64
import numpy as np

from fastapi import FastAPI, APIRouter, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize


app = FastAPI(
    title="Mask R-CNN API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoculeCount(BaseModel):
    photo: bytes
    count: int

class ModelConfig(Config):
    NAME = "model_config"
    BACKBONE = "resnet50"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 2 + 1

models = {}

@app.get("/")
def read_root():
    return {"Hello": "From Mask R-CNN API!"}

@app.on_event("startup")
def initialize():
    try:
        print("Initializing model...")
        models['mask-rcnn'] = modellib.MaskRCNN(mode="inference", model_dir=os.getcwd(), config=ModelConfig())
        models['mask-rcnn'].load_weights('./mask_rcnn_locules_0003.h5', by_name=True)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error initializing the model: {e}")
        raise RuntimeError(f"Error initializing the model: {e}")

# Endpoint to predict and visualize locules
@app.post("/predict/", response_model=LoculeCount)
async def count_locules(input: UploadFile) -> LoculeCount:
    try:
        # Read and validate the uploaded image
        image_data = await input.read()
        if not image_data:
            raise ValueError("Uploaded file is empty.")
        
        photo_stream = io.BytesIO(image_data)
        image = Image.open(photo_stream)
        image_array = np.array(image)
        
        # Handle grayscale and RGBA images
        if image_array.ndim == 2:  # Grayscale
            image_array = np.stack((image_array,) * 3, axis=-1)
        elif image_array.shape[2] == 4:  # RGBA
            image_array = image_array[:, :, :3]
        
        # Ensure the model is loaded
        if 'mask-rcnn' not in models or models['mask-rcnn'] is None:
            raise RuntimeError("Model not loaded. Please check the initialization process.")
        
        # Run Mask R-CNN detection
        results = models['mask-rcnn'].detect([image_array], verbose=1)
        if not results or 'masks' not in results[0]:
            raise ValueError("No masks detected in the image.")
        
        # Count locules
        locule_count = results[0]['masks'].shape[2]
        
        # Visualize the results on the image
        _, np_array_image = visualize.display_instances(
            image_array,
            boxes=results[0]['rois'],
            masks=results[0]['masks'],
            class_ids=results[0]['class_ids'],
            class_names=['BG', 'locule'],
            scores=results[0]['scores'],
        )
        
        # Handle RGBA images
        if np_array_image.shape[2] == 4:
            np_array_image = np_array_image[:, :, :3]
            
        masked_image = Image.fromarray(np_array_image)

        # Convert the visualized image to base64
        img_byte_arr = io.BytesIO()
        masked_image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        photo_base64 = base64.b64encode(img_byte_arr).decode('utf-8')

        # Return the response
        return LoculeCount(photo=photo_base64, count=locule_count)
    
    except Exception as e:
        error_message = traceback.format_exc()
        print("Error:", error_message)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing image: {e}",
        )
