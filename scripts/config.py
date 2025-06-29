import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings:

    IMAGE_DIR: Path = PROJECT_ROOT / "assets" / "images"
    IMAGE_NAME: str = "image.jpg"
    IMAGE_PATH: Path = IMAGE_DIR / IMAGE_NAME
    IMAGE_SAVE_PATH: Path = IMAGE_DIR / "segmented_image.jpg"

    MODEL_DIR: Path = PROJECT_ROOT / "assets" / "models"
    MODEL_NAME: str = "segmentation_model.h5"
    MODEL_SAVE_PATH: Path = MODEL_DIR / MODEL_NAME

settings = Settings()