from fastapi import FastAPI, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import base64
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI(title="YOLOv9 API", root_path="/api/yolo-inference")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic model for response
class LoculeCount(BaseModel):
    photo: str  # Base64-encoded image
    count: int

@app.get("/")
def read_root():
    return {"message": "Welcome to YOLOv9 API!"}

# Load YOLO model at startup
@app.on_event("startup")
async def load_model():
    global model
    try:
        model = YOLO("yolo_model.pt")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load YOLO model: {e}",
        )

@app.post("/predict/", response_model=LoculeCount)
async def count_locules(input: UploadFile) -> LoculeCount:
    try:
        # Read and convert image
        image_bytes = await input.read()
        image_stream = io.BytesIO(image_bytes)
        image = Image.open(image_stream).convert("RGB")
        
        # Convert image to numpy array for processing
        image_array = np.array(image)
        
        # Convert image to OpenCV format for YOLO (RGB to BGR)
        opencv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Run YOLO model inference
        result = model(opencv_image, conf=0.8)

        # Extract bounding boxes and confidence scores
        boxes = result[0].boxes.xyxy.cpu().numpy()
        confidences = result[0].boxes.conf.cpu().numpy()
        masks = result[0].masks
        locule_count = len(boxes)

        # Define colors for visualization
        predefined_colors = [
            (255, 0, 0),   # Red
            (0, 0, 255),   # Blue
            (12, 133, 22), # Green
            (128, 0, 128), # Purple
            (150, 64, 2)   # Yellow
        ]
        colors = [predefined_colors[i % len(predefined_colors)] for i in range(locule_count)]

        # Create an overlay for segmentation masks
        overlay = np.zeros_like(image_array, dtype=np.uint8)

        # Process masks if they exist
        if masks is not None:
            # Get the original shape from the masks
            orig_shape = masks.orig_shape  # Use the original shape from the masks object
            mask_tensor = masks.data.cpu()
            num_masks = mask_tensor.shape[0]

            for i in range(num_masks):
                mask = mask_tensor[i].numpy()  # Convert tensor to NumPy
                mask = (mask > 0.5).astype(np.uint8)  # Binarize mask

                # Resize mask to match the original image dimensions
                mask = cv2.resize(mask, (image.width, image.height), interpolation=cv2.INTER_LINEAR)

                # Apply Gaussian Blur to soften edges
                mask = cv2.GaussianBlur(mask, (5, 5), 0)

                # Create a 3-channel colored mask
                color = colors[i]
                for c in range(3):  # Apply color to each channel
                    overlay[:, :, c] += mask * color[c]

            # Blend the overlay with the original image
            alpha = 0.6  # Transparency factor
            image_array = cv2.addWeighted(image_array, 1, overlay, alpha, 0)

        # Function to draw a dotted bounding box
        def draw_dotted_rectangle(img, pt1, pt2, color, gap=5):
            x1, y1 = pt1
            x2, y2 = pt2

            # Draw dotted lines for each side
            for x in range(x1, x2, gap * 2):
                cv2.line(img, (x, y1), (x + gap, y1), color, thickness=1)

            for x in range(x1, x2, gap * 2):
                cv2.line(img, (x, y2), (x + gap, y2), color, thickness=1)

            for y in range(y1, y2, gap * 2):
                cv2.line(img, (x1, y), (x1, y + gap), color, thickness=1)

            for y in range(y1, y2, gap * 2):
                cv2.line(img, (x2, y), (x2, y + gap), color, thickness=1)

        # Draw bounding boxes and labels
        for box, conf, color in zip(boxes, confidences, colors):
            x1, y1, x2, y2 = map(int, box)

            # Draw the dotted bounding box
            draw_dotted_rectangle(image_array, (x1, y1), (x2, y2), color)

            # Format confidence score label
            label = f"Locule {conf:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

            # Define text position (top-right of bounding box)
            text_x = x2 - text_size[0] - 5
            text_y = y1 - 5 if y1 - 5 > 10 else y1 + 15

            # Background rectangle for text
            text_bg_x1, text_bg_y1 = text_x - 2, text_y - text_size[1] - 2
            text_bg_x2, text_bg_y2 = text_x + text_size[0] + 4, text_y + 4

            # Draw label background
            cv2.rectangle(image_array, (text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2), color, thickness=-1)

            # Draw label text
            cv2.putText(image_array, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Convert processed image back to high-quality JPEG format
        image_pil = Image.fromarray(image_array)  # No need for BGR2RGB conversion as we worked in RGB
        image_stream = io.BytesIO()
        image_pil.save(image_stream, format="JPEG", quality=95, optimize=True)  # High-quality JPEG
        encoded_image = base64.b64encode(image_stream.getvalue()).decode()

        return LoculeCount(count=locule_count, photo=encoded_image)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing image: {e}",
        )


# from fastapi import FastAPI, APIRouter, UploadFile, HTTPException, status
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import io
# import base64
# import cv2
# import numpy as np
# from PIL import Image
# from ultralytics import YOLO

# app = FastAPI(
#     title="YOLOv9 API",
#     root_path="/api/yolo-inference",
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Limit this for production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class LoculeCount(BaseModel):
#     photo: bytes
#     count: int

# @app.get("/")
# def read_root():
#     return {"Hello": "From YOLOv9 API!"}

# @app.on_event("startup")
# async def load_model():
#     try:
#         global model
#         model = YOLO("yolo_model.pt")
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to load model: {e}",
#         )

# @app.post("/predict/", response_model=LoculeCount)
# async def count_locules(input: UploadFile) -> LoculeCount:
#     locule_count = 0
#     try:
#         image = await input.read()
#         photo_stream = io.BytesIO(image)
#         image = Image.open(photo_stream)
#         image_array = np.array(image)

#         if image_array.ndim == 2:  # Grayscale to RGB
#             image_array = np.stack((image_array,) * 3, axis=-1)
#         elif image_array.shape[2] == 4:  # RGBA to RGB
#             image_array = image_array[:, :, :3]

#         image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

#         results = model(image_array, conf=0.8)
#         objects = results[0].boxes
#         detected_durian = objects.cls.tolist()
#         locule_count = len(detected_durian)

#         image_with_masks = results[0].plot()
#         success, encoded_image = cv2.imencode(".jpg", image_with_masks)
        
#         if not success:
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail="Failed to encode image",
#             )

        
#         return LoculeCount(
#             count=locule_count,
#             photo=base64.b64encode(encoded_image).decode(),
#         )

#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=f"Error processing image: {e}",
#         )

