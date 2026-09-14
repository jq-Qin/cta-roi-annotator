from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


RASTER_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageSource:
    path: Path
    kind: str


def natural_sort_key(value: str | Path) -> tuple[object, ...]:
    """Return a case-insensitive key where digit groups sort numerically."""
    text = Path(value).name if isinstance(value, Path) else value
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", text))


def _is_dicom(path: Path) -> bool:
    if path.suffix.casefold() == ".dcm":
        return True
    try:
        from pydicom.misc import is_dicom

        return bool(is_dicom(path))
    except (ImportError, OSError):
        return False


def discover_images(image_dir: str | Path) -> list[ImageSource]:
    directory = Path(image_dir)
    if not directory.is_dir():
        raise ValueError(f"Image directory does not exist: {directory}")

    sources: list[ImageSource] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        suffix = path.suffix.casefold()
        if suffix in RASTER_SUFFIXES:
            sources.append(ImageSource(path=path, kind="raster"))
        elif _is_dicom(path):
            sources.append(ImageSource(path=path, kind="dicom"))
    sources.sort(key=lambda source: natural_sort_key(source.path.name))
    return sources


def _decode_raster(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape {image.shape}: {path}")
    return image


def _first_number(value: object) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, (str, bytes)):
            return float(value)
        return float(value[0])  # type: ignore[index]
    except (TypeError, ValueError, IndexError):
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None


def _decode_dicom(path: Path) -> tuple[np.ndarray, tuple[float, float] | None]:
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError("DICOM input requires pydicom. Install requirements.txt.") from exc

    dataset = pydicom.dcmread(str(path), force=True)
    try:
        pixels = np.asarray(dataset.pixel_array)
    except Exception as exc:
        raise ValueError(f"Cannot decode DICOM pixel data: {path}: {exc}") from exc
    if pixels.ndim != 2:
        raise ValueError(f"Only single-frame 2D DICOM is supported, got {pixels.shape}: {path}")

    slope = float(getattr(dataset, "RescaleSlope", 1.0))
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
    image = pixels.astype(np.float32) * slope + intercept
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        image = image.max() + image.min() - image

    center = _first_number(getattr(dataset, "WindowCenter", None))
    width = _first_number(getattr(dataset, "WindowWidth", None))
    window = (center, width) if center is not None and width is not None and width > 0 else None
    return image, window


def read_source(source: ImageSource) -> tuple[np.ndarray, tuple[float, float] | None]:
    if source.kind == "dicom":
        return _decode_dicom(source.path)
    return _decode_raster(source.path), None


def make_display_image(
    image: np.ndarray,
    window: tuple[float, float] | None = None,
) -> np.ndarray:
    """Convert source pixels to uint8 for display only; never mutates source data."""
    if image.ndim == 3:
        if image.shape[2] == 4:
            color = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        elif image.shape[2] == 3:
            color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported channel count: {image.shape}")
        if color.dtype == np.uint8:
            return color
        color_values = color.astype(np.float32)
        finite = color_values[np.isfinite(color_values)]
        if finite.size == 0 or float(finite.max()) <= float(finite.min()):
            return np.zeros(color.shape, dtype=np.uint8)
        low, high = float(finite.min()), float(finite.max())
        return np.clip((color_values - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)

    values = image.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    if window is not None:
        center, width = window
        low, high = center - width / 2.0, center + width / 2.0
    else:
        low, high = np.percentile(finite, (1.0, 99.0))
        if high <= low:
            low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.clip((values - low) * (255.0 / (high - low)), 0, 255)
    return scaled.astype(np.uint8)


def read_mask(path: str | Path) -> np.ndarray:
    mask_path = Path(path)
    encoded = np.fromfile(mask_path, dtype=np.uint8)
    mask = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot decode mask: {mask_path}")
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


def _write_binary_mask_atomic(path: Path, mask: np.ndarray) -> None:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    success, encoded = cv2.imencode(".png", binary)
    if not success:
        raise OSError(f"Failed to encode mask: {path}")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_bytes(encoded.tobytes())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class AnnotationSession:
    """Owns source-to-mask mapping and all non-UI annotation state."""

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        *,
        max_undo: int = 30,
    ) -> None:
        self.image_dir = Path(image_dir).resolve()
        self.mask_dir = Path(mask_dir).resolve()
        if self.image_dir == self.mask_dir:
            raise ValueError("Image and mask directories must be different.")
        self.sources = discover_images(self.image_dir)
        if not self.sources:
            raise ValueError(f"No supported images found in: {self.image_dir}")
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.max_undo = max(1, int(max_undo))
        self.index = 0
        self.current_image = np.empty((0, 0), dtype=np.uint8)
        self.display_image = np.empty((0, 0), dtype=np.uint8)
        self.display_window: tuple[float, float] | None = None
        self.mask = np.empty((0, 0), dtype=np.uint8)
        self.dirty = False
        self._undo: list[np.ndarray] = []
        self._load_index(0)

    @property
    def current_source(self) -> ImageSource:
        return self.sources[self.index]

    @property
    def current_mask_path(self) -> Path:
        # PNG encoding is used internally, while the exact source filename is retained.
        return self.mask_dir / self.current_source.path.name

    @property
    def progress_text(self) -> str:
        return f"{self.index + 1} / {len(self.sources)}"

    @property
    def is_labeled(self) -> bool:
        return self.current_mask_path.is_file()

    def _load_index(self, index: int) -> None:
        self.index = index
        self.current_image, self.display_window = read_source(self.current_source)
        self.display_image = make_display_image(self.current_image, self.display_window)
        expected_shape = self.current_image.shape[:2]
        mask_path = self.current_mask_path
        if mask_path.is_file():
            mask = read_mask(mask_path)
            if mask.shape != expected_shape:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match image shape {expected_shape}: {mask_path}"
                )
            self.mask = mask
        else:
            self.mask = np.zeros(expected_shape, dtype=np.uint8)
        self._undo.clear()
        self.dirty = False

    def _remember(self) -> None:
        self._undo.append(self.mask.copy())
        if len(self._undo) > self.max_undo:
            del self._undo[0]

    def add_polygon(self, points: Sequence[tuple[int, int]]) -> None:
        if len(points) < 3:
            raise ValueError("A polygon requires at least three points.")
        polygon = np.asarray(points, dtype=np.int32)
        self._remember()
        cv2.fillPoly(self.mask, [polygon], color=255)
        self.dirty = True

    def add_circle(self, center: tuple[int, int], radius: int) -> None:
        if radius < 1:
            raise ValueError("Circle radius must be at least 1.")
        self._remember()
        cv2.circle(self.mask, center, int(radius), 255, thickness=-1)
        self.dirty = True

    def apply_brush_stroke(
        self,
        points: Sequence[tuple[int, int]],
        *,
        radius: int,
        erase: bool = False,
    ) -> None:
        if not points:
            raise ValueError("A brush stroke requires at least one point.")
        if radius < 1:
            raise ValueError("Brush radius must be at least 1.")
        self._remember()
        color = 0 if erase else 255
        for point in points:
            cv2.circle(self.mask, point, radius, color, thickness=-1)
        for start, end in zip(points, points[1:]):
            cv2.line(self.mask, start, end, color, thickness=radius * 2)
        self.dirty = True

    def begin_external_edit(self) -> np.ndarray:
        return self.mask.copy()

    def commit_external_edit(self, before: np.ndarray) -> bool:
        if before.shape != self.mask.shape:
            raise ValueError("Edit snapshot shape does not match current mask.")
        if np.array_equal(before, self.mask):
            return False
        self._undo.append(before.copy())
        if len(self._undo) > self.max_undo:
            del self._undo[0]
        self.mask[:] = np.where(self.mask > 0, 255, 0).astype(np.uint8)
        self.dirty = True
        return True

    def clear(self) -> None:
        if not np.any(self.mask):
            return
        self._remember()
        self.mask.fill(0)
        self.dirty = True

    def undo(self) -> bool:
        if not self._undo:
            return False
        self.mask = self._undo.pop()
        self.dirty = True
        return True

    def copy_previous_mask(self) -> None:
        if self.index == 0:
            raise ValueError("There is no previous image.")
        previous_path = self.mask_dir / self.sources[self.index - 1].path.name
        if previous_path.is_file():
            previous = read_mask(previous_path)
        else:
            previous_image, _ = read_source(self.sources[self.index - 1])
            previous = np.zeros(previous_image.shape[:2], dtype=np.uint8)
        if previous.shape != self.mask.shape:
            raise ValueError(
                f"Previous mask shape {previous.shape} does not match current image shape {self.mask.shape}."
            )
        self._remember()
        self.mask = previous.copy()
        self.dirty = True

    def save(self) -> Path:
        expected_shape = self.current_image.shape[:2]
        if self.mask.shape != expected_shape:
            raise ValueError(f"Mask shape {self.mask.shape} does not match image shape {expected_shape}.")
        if self.mask.dtype != np.uint8 or not set(np.unique(self.mask)).issubset({0, 255}):
            self.mask = np.where(self.mask > 0, 255, 0).astype(np.uint8)
        path = self.current_mask_path
        _write_binary_mask_atomic(path, self.mask)
        self.dirty = False
        return path

    def navigate(self, delta: int) -> bool:
        target = min(max(self.index + int(delta), 0), len(self.sources) - 1)
        return self.navigate_to(target)

    def navigate_to(self, index: int) -> bool:
        if not 0 <= index < len(self.sources):
            raise IndexError(f"Slice index out of range: {index}")
        if index == self.index:
            return False
        if self.dirty:
            self.save()
        self._load_index(index)
        return True

    def close(self) -> None:
        if self.dirty:
            self.save()
