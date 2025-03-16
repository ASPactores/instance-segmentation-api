from pydantic import BaseModel

class LoculeCount(BaseModel):
    photo: str  # Base64-encoded image
    count: int