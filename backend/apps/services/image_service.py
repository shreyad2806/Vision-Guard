"""Handles image file validation, storage, and reading."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import cv2
import numpy as np

from apps.core.config import settings

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_SIZE_BYTES = settings.max_upload_size_mb * 1024 * 1024


def validate_extension(filename: str) -> bool:
    """Return True if the file extension is in the allowed set."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def validate_size(data: bytes) -> bool:
    """Return True if the upload is within the size limit."""
    return len(data) <= MAX_SIZE_BYTES


def save_upload(data: bytes, original_filename: str) -> tuple[str, str]:
    """Persist the upload to disk and return (stored_name, absolute_path).

    The stored name uses a UUID prefix to avoid collisions.
    """
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(original_filename).suffix.lower()
    stored_name = f"{uuid.uuid4().hex}{ext}"
    dest = upload_dir / stored_name
    dest.write_bytes(data)
    return stored_name, str(dest.resolve())


def read_image(image_path: str) -> np.ndarray | None:
    """Read an image from disk into a BGR numpy array.

    Returns None if the file cannot be decoded.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return img
