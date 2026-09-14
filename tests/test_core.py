from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from cta_annotator.core import (
    AnnotationSession,
    discover_images,
    natural_sort_key,
    read_mask,
)


def _write_image(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(path.suffix or ".png", image)
    assert success
    path.write_bytes(encoded.tobytes())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def image_dirs(tmp_path: Path) -> tuple[Path, Path]:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    for index, name in enumerate(("CTA_10.png", "CTA_2.png", "CTA_1.png")):
        image = np.full((32, 48), 30 + index, dtype=np.uint8)
        _write_image(image_dir / name, image)
    return image_dir, mask_dir


def test_natural_sort_and_supported_file_discovery(image_dirs: tuple[Path, Path]) -> None:
    image_dir, _ = image_dirs
    (image_dir / "notes.txt").write_text("not an image", encoding="utf-8")

    sources = discover_images(image_dir)

    assert [source.path.name for source in sources] == [
        "CTA_1.png",
        "CTA_2.png",
        "CTA_10.png",
    ]
    assert natural_sort_key("slice12.png") > natural_sort_key("slice2.png")


def test_polygon_save_is_exact_binary_and_does_not_modify_source(
    image_dirs: tuple[Path, Path],
) -> None:
    image_dir, mask_dir = image_dirs
    original_hashes = {path.name: _sha256(path) for path in image_dir.glob("*.png")}
    session = AnnotationSession(image_dir, mask_dir)

    session.add_polygon([(4, 4), (30, 4), (30, 20), (4, 20)])
    saved_path = session.save()
    saved = read_mask(saved_path)

    assert saved_path.name == session.current_source.path.name
    assert saved.dtype == np.uint8
    assert saved.shape == session.current_image.shape[:2]
    assert set(np.unique(saved)).issubset({0, 255})
    assert saved[10, 10] == 255
    assert saved[0, 0] == 0
    assert {path.name: _sha256(path) for path in image_dir.glob("*.png")} == original_hashes


def test_existing_mask_reloads_and_can_be_edited(image_dirs: tuple[Path, Path]) -> None:
    image_dir, mask_dir = image_dirs
    session = AnnotationSession(image_dir, mask_dir)
    session.add_polygon([(3, 3), (20, 3), (20, 15), (3, 15)])
    session.save()

    reopened = AnnotationSession(image_dir, mask_dir)
    assert reopened.mask[8, 8] == 255
    reopened.apply_brush_stroke([(8, 8), (8, 8)], radius=4, erase=True)
    reopened.save()

    edited = read_mask(reopened.current_mask_path)
    assert edited[8, 8] == 0
    assert set(np.unique(edited)).issubset({0, 255})


def test_copy_previous_undo_and_clear(image_dirs: tuple[Path, Path]) -> None:
    image_dir, mask_dir = image_dirs
    session = AnnotationSession(image_dir, mask_dir)
    session.add_polygon([(5, 5), (25, 5), (25, 20), (5, 20)])
    expected = session.mask.copy()
    session.navigate(1)

    session.copy_previous_mask()
    assert np.array_equal(session.mask, expected)
    copied = session.mask.copy()

    session.clear()
    assert not np.any(session.mask)
    assert session.undo()
    assert np.array_equal(session.mask, copied)


def test_navigation_auto_saves_to_correct_filename_without_cross_talk(
    image_dirs: tuple[Path, Path],
) -> None:
    image_dir, mask_dir = image_dirs
    session = AnnotationSession(image_dir, mask_dir)
    first_name = session.current_source.path.name
    session.apply_brush_stroke([(7, 7)], radius=2)
    first_expected = session.mask.copy()

    session.navigate(1)
    second_name = session.current_source.path.name
    assert (mask_dir / first_name).is_file()
    assert np.array_equal(read_mask(mask_dir / first_name), first_expected)
    assert not np.any(session.mask)

    session.apply_brush_stroke([(15, 15)], radius=3)
    second_expected = session.mask.copy()
    session.navigate(1)
    assert np.array_equal(read_mask(mask_dir / second_name), second_expected)
    assert not np.array_equal(read_mask(mask_dir / first_name), second_expected)


def test_rejects_same_image_and_mask_directory(image_dirs: tuple[Path, Path]) -> None:
    image_dir, _ = image_dirs

    with pytest.raises(ValueError, match="different"):
        AnnotationSession(image_dir, image_dir)


def test_rejects_existing_mask_with_wrong_shape(image_dirs: tuple[Path, Path]) -> None:
    image_dir, mask_dir = image_dirs
    mask_dir.mkdir()
    _write_image(mask_dir / "CTA_1.png", np.zeros((3, 4), dtype=np.uint8))

    with pytest.raises(ValueError, match="shape"):
        AnnotationSession(image_dir, mask_dir)


def test_extensionless_dicom_loads_and_keeps_exact_filename(tmp_path: Path) -> None:
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    image_dir = tmp_path / "dicom"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    path = image_dir / "I1000000"
    pixels = np.arange(20 * 24, dtype=np.int16).reshape(20, 24)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.RescaleSlope = 1
    dataset.RescaleIntercept = -1024
    dataset.WindowCenter = 100
    dataset.WindowWidth = 800
    dataset.PixelData = pixels.tobytes()
    dataset.save_as(path)

    session = AnnotationSession(image_dir, mask_dir)
    assert session.current_image.shape == pixels.shape
    assert session.current_image.dtype == np.float32
    assert session.display_image.dtype == np.uint8
    session.apply_brush_stroke([(10, 10)], radius=2)
    saved_path = session.save()

    assert saved_path.name == "I1000000"
    assert read_mask(saved_path).shape == pixels.shape


def test_existing_mask_uses_safe_binary_threshold(tmp_path: Path) -> None:
    path = tmp_path / "mask.jpg"
    values = np.array([[0, 1, 127, 128, 254, 255]], dtype=np.uint8)
    success, encoded = cv2.imencode(".png", values)
    assert success
    path.write_bytes(encoded.tobytes())

    loaded = read_mask(path)

    assert loaded.tolist() == [[0, 0, 0, 255, 255, 255]]


def test_circle_annotation_is_filled_binary_and_undoable(
    image_dirs: tuple[Path, Path],
) -> None:
    image_dir, mask_dir = image_dirs
    session = AnnotationSession(image_dir, mask_dir)

    session.add_circle(center=(20, 16), radius=7)

    assert session.mask[16, 20] == 255
    assert session.mask[16, 27] == 255
    assert session.mask[0, 0] == 0
    assert set(np.unique(session.mask)) == {0, 255}
    assert session.undo()
    assert not np.any(session.mask)


def test_circle_rejects_non_positive_radius(image_dirs: tuple[Path, Path]) -> None:
    image_dir, mask_dir = image_dirs
    session = AnnotationSession(image_dir, mask_dir)

    with pytest.raises(ValueError, match="radius"):
        session.add_circle(center=(10, 10), radius=0)


def test_navigate_to_non_contiguous_slice_auto_saves_correct_image(
    image_dirs: tuple[Path, Path],
) -> None:
    image_dir, mask_dir = image_dirs
    session = AnnotationSession(image_dir, mask_dir)
    first_name = session.current_source.path.name
    session.add_circle(center=(12, 12), radius=4)
    first_mask = session.mask.copy()

    assert session.navigate_to(2)
    assert session.index == 2
    assert np.array_equal(read_mask(mask_dir / first_name), first_mask)
    assert not np.any(session.mask)

    with pytest.raises(IndexError, match="index"):
        session.navigate_to(99)
