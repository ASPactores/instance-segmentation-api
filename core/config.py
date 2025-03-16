from typing import List

class Settings:
    # API settings
    APP_TITLE: str = "YOLOv9 API"
    ROOT_PATH: str = "/api"
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]  # Change this for production
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]
    
    # Model settings
    MODEL_PATH: str = "yolo_model.pt"
    CONFIDENCE_THRESHOLD: float = 0.8
    
    # Visualization settings
    MASK_ALPHA: float = 0.6
    IMAGE_QUALITY: int = 95
    
    # Predefined colors for visualization
    COLORS: List[tuple] = [
        (255, 0, 0),   # Red
        (0, 0, 255),   # Blue
        (12, 133, 22), # Green
        (128, 0, 128), # Purple
        (150, 64, 2)   # Yellow
    ]
    
    class Config:
        env_file = ".env"

settings = Settings()
