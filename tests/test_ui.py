from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from cta_annotator.ui import MainWindow


def test_main_window_builds_and_tracks_session_state(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "new_masks"
    image_dir.mkdir()
    for index in range(3):
        image = np.full((40, 50), index * 40, dtype=np.uint8)
        success, encoded = cv2.imencode(".png", image)
        assert success
        (image_dir / f"slice{index}.png").write_bytes(encoded.tobytes())

    app = QApplication.instance() or QApplication([])
    window = MainWindow(image_dir=image_dir, mask_dir=mask_dir)
    assert window.session is not None
    assert window.progress_label.text() == "1 / 3"
    assert window.file_label.text() == "slice0.png"
    assert window.slice_list.count() == 3
    assert mask_dir.is_dir()

    window.session.add_polygon([(2, 2), (20, 2), (20, 20), (2, 20)])
    window.update_status()
    assert "Modified" in window.state_label.text()
    window.navigate(1)
    assert window.file_label.text() == "slice1.png"
    assert (mask_dir / "slice0.png").is_file()

    before = window.session.begin_external_edit()
    cv2.circle(window.session.mask, (10, 10), 3, 255, thickness=-1)
    window.canvas._painting = True
    window.canvas._stroke_before = before
    window.navigate(1)
    assert window.file_label.text() == "slice2.png"
    assert (mask_dir / "slice1.png").is_file()

    window.go_to_index(0)
    assert window.file_label.text() == "slice0.png"
    window.close()
    app.processEvents()
