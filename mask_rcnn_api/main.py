import io
import base64
import cv2
import numpy as np

from fastapi import FastAPI, APIRouter, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO


app = FastAPI(
    title="Mask R-CNN API",
    root_path="/api/mask-rcnn-inference",
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

@app.get("/")
def read_root():
    return {"Hello": "From Mask R-CNN API!"}

@app.post("/predict/", response_model=LoculeCount)
async def count_locules(input: UploadFile) -> LoculeCount:
    pass