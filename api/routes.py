import io
import base64
import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, HTTPException, status, Depends
from models.schemas import LoculeCount
from services.yolo_service import yolo_service
from utils.visualization import process_yolo_result
from core.config import settings

router = APIRouter()

@router.get("/")
def read_root():
    return {"message": f"Welcome to {settings.APP_TITLE}!"}

@router.on_event("startup")
async def startup_event():
    """Load YOLO model on startup."""
    try:
        yolo_service.load_model()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load YOLO model: {e}",
        )

@router.post("/predict/", response_model=LoculeCount)
async def count_locules(input: UploadFile) -> LoculeCount:
    """
    Process an uploaded image to count locules using YOLO model.
    
    Returns:
        LoculeCount: Object containing the count of locules and 
                     the processed image with visualizations
    """
    try:
        # Read and convert image
        image_bytes = await input.read()
        image_stream = io.BytesIO(image_bytes)
        image = Image.open(image_stream).convert("RGB")
        
        # Convert image to numpy array for processing
        image_array = np.array(image)
        
        # Run YOLO model inference
        result = yolo_service.predict(image_array)
        
        # Process results and visualize
        processed_image, locule_count = process_yolo_result(image_array, result)
        
        # Convert processed image back to high-quality JPEG format
        image_pil = Image.fromarray(processed_image)
        image_stream = io.BytesIO()
        image_pil.save(image_stream, format="JPEG", quality=settings.IMAGE_QUALITY, optimize=True)
        encoded_image = base64.b64encode(image_stream.getvalue()).decode()
        
        return LoculeCount(count=locule_count, photo=encoded_image)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing image: {e}",
        )