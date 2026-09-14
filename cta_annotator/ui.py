from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .core import AnnotationSession


def _array_to_qimage(image: np.ndarray) -> QImage:
    contiguous = np.ascontiguousarray(image)
    if contiguous.ndim == 2:
        height, width = contiguous.shape
        qimage = QImage(contiguous.data, width, height, contiguous.strides[0], QImage.Format_Grayscale8)
    elif contiguous.shape[2] == 3:
        height, width, _ = contiguous.shape
        qimage = QImage(contiguous.data, width, height, contiguous.strides[0], QImage.Format_RGB888)
    elif contiguous.shape[2] == 4:
        height, width, _ = contiguous.shape
        qimage = QImage(contiguous.data, width, height, contiguous.strides[0], QImage.Format_RGBA8888)
    else:
        raise ValueError(f"Unsupported display image shape: {contiguous.shape}")
    return qimage.copy()


def _mask_overlay(mask: np.ndarray, opacity: int) -> QImage:
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    selected = mask > 0
    rgba[selected, 0] = 255
    rgba[selected, 1] = 35
    rgba[selected, 2] = 35
    rgba[selected, 3] = opacity
    return _array_to_qimage(rgba)


class AnnotationCanvas(QGraphicsView):
    slice_requested = Signal(int)
    mask_changed = Signal()
    message = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#171717"))
        self.setRenderHints(self.renderHints())
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self._base_item = QGraphicsPixmapItem()
        self._overlay_item = QGraphicsPixmapItem()
        self._overlay_item.setZValue(1)
        self.scene().addItem(self._base_item)
        self.scene().addItem(self._overlay_item)
        self._polygon_item = QGraphicsPathItem()
        self._polygon_item.setPen(QPen(QColor(0, 255, 255), 2))
        self._polygon_item.setZValue(2)
        self.scene().addItem(self._polygon_item)
        self._circle_item = QGraphicsEllipseItem()
        self._circle_item.setPen(QPen(QColor(0, 255, 255), 2))
        self._circle_item.setZValue(2)
        self._circle_item.setVisible(False)
        self.scene().addItem(self._circle_item)
        self.session: AnnotationSession | None = None
        self.mode = "polygon"
        self.brush_radius = 10
        self.overlay_visible = True
        self.overlay_opacity = 105
        self._polygon_points: list[tuple[int, int]] = []
        self._freehand_active = False
        self._circle_center: tuple[int, int] | None = None
        self._circle_radius = 0
        self._painting = False
        self._stroke_before: np.ndarray | None = None
        self._last_point: tuple[int, int] | None = None
        self._panning = False
        self._pan_position = QPoint()

    def set_session(self, session: AnnotationSession) -> None:
        self.session = session
        self.cancel_polygon()
        self.refresh_image(fit=True)

    def set_mode(self, mode: str) -> None:
        if mode not in {"polygon", "circle", "freehand", "brush", "eraser"}:
            raise ValueError(f"Unknown tool mode: {mode}")
        self.cancel_polygon()
        self.mode = mode

    def refresh_image(self, *, fit: bool = False) -> None:
        if self.session is None:
            return
        base = QPixmap.fromImage(_array_to_qimage(self.session.display_image))
        self._base_item.setPixmap(base)
        self.scene().setSceneRect(0, 0, base.width(), base.height())
        self.refresh_overlay()
        if fit:
            self.resetTransform()
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)

    def refresh_overlay(self) -> None:
        if self.session is None:
            return
        overlay = QPixmap.fromImage(_mask_overlay(self.session.mask, self.overlay_opacity))
        self._overlay_item.setPixmap(overlay)
        self._overlay_item.setVisible(self.overlay_visible)

    def toggle_overlay(self) -> None:
        self.overlay_visible = not self.overlay_visible
        self._overlay_item.setVisible(self.overlay_visible)

    def zoom(self, factor: float) -> None:
        current = self.transform().m11()
        target = current * factor
        if 0.05 <= target <= 40.0:
            self.scale(factor, factor)

    def _image_point(self, position: QPoint) -> tuple[int, int] | None:
        if self.session is None:
            return None
        scene_point = self.mapToScene(position)
        x, y = int(round(scene_point.x())), int(round(scene_point.y()))
        height, width = self.session.mask.shape
        if 0 <= x < width and 0 <= y < height:
            return x, y
        return None

    def _draw_polygon_preview(self) -> None:
        path = QPainterPath()
        if self._polygon_points:
            first = self._polygon_points[0]
            path.moveTo(*first)
            for point in self._polygon_points[1:]:
                path.lineTo(*point)
        self._polygon_item.setPath(path)

    def close_polygon(self) -> None:
        if self.session is None or not self._polygon_points:
            return
        points = self._polygon_points
        self._polygon_points = []
        self._freehand_active = False
        self._draw_polygon_preview()
        if len(points) < 3:
            self.message.emit("Polygon needs at least 3 points")
            return
        self.session.add_polygon(points)
        self.refresh_overlay()
        self.mask_changed.emit()

    def cancel_polygon(self) -> None:
        self._polygon_points = []
        self._freehand_active = False
        self._circle_center = None
        self._circle_radius = 0
        self._circle_item.setVisible(False)
        self._draw_polygon_preview()

    def finish_stroke(self) -> None:
        """Commit an in-progress stroke before any slice state can change."""
        if not self._painting or self.session is None:
            return
        self._painting = False
        if self._stroke_before is not None:
            self.session.commit_external_edit(self._stroke_before)
        self._stroke_before = None
        self._last_point = None
        self.mask_changed.emit()

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.modifiers() & Qt.ControlModifier:
            self.zoom(1.2 if event.angleDelta().y() > 0 else 1 / 1.2)
        else:
            self.slice_requested.emit(-1 if event.angleDelta().y() > 0 else 1)
        event.accept()

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_position = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        point = self._image_point(event.position().toPoint())
        if event.button() == Qt.LeftButton and point is not None and self.session is not None:
            if self.mode == "polygon":
                self._polygon_points.append(point)
                self._draw_polygon_preview()
            elif self.mode == "circle":
                self._circle_center = point
                self._circle_radius = 0
                self._circle_item.setRect(point[0], point[1], 0, 0)
                self._circle_item.setVisible(True)
            elif self.mode == "freehand":
                self._polygon_points = [point]
                self._freehand_active = True
                self._draw_polygon_preview()
            else:
                self._painting = True
                self._stroke_before = self.session.begin_external_edit()
                self._last_point = point
                color = 0 if self.mode == "eraser" else 255
                cv2.circle(self.session.mask, point, self.brush_radius, color, thickness=-1)
                self.refresh_overlay()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.LeftButton and self.mode == "polygon":
            self.close_polygon()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._panning:
            current = event.position().toPoint()
            delta = current - self._pan_position
            self._pan_position = current
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        if self._circle_center is not None:
            point = self._image_point(event.position().toPoint())
            if point is not None:
                center_x, center_y = self._circle_center
                self._circle_radius = max(
                    1,
                    int(round(math.hypot(point[0] - center_x, point[1] - center_y))),
                )
                radius = self._circle_radius
                self._circle_item.setRect(
                    center_x - radius,
                    center_y - radius,
                    radius * 2,
                    radius * 2,
                )
            event.accept()
            return
        if self._freehand_active:
            point = self._image_point(event.position().toPoint())
            if point is not None and point != self._polygon_points[-1]:
                self._polygon_points.append(point)
                self._draw_polygon_preview()
            event.accept()
            return
        if self._painting and self.session is not None and self._last_point is not None:
            point = self._image_point(event.position().toPoint())
            if point is not None:
                color = 0 if self.mode == "eraser" else 255
                cv2.line(
                    self.session.mask,
                    self._last_point,
                    point,
                    color,
                    thickness=self.brush_radius * 2,
                )
                cv2.circle(self.session.mask, point, self.brush_radius, color, thickness=-1)
                self._last_point = point
                self.refresh_overlay()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self.unsetCursor()
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._circle_center is not None:
            center = self._circle_center
            radius = self._circle_radius
            self._circle_center = None
            self._circle_radius = 0
            self._circle_item.setVisible(False)
            if self.session is not None and radius >= 1:
                self.session.add_circle(center, radius)
                self.refresh_overlay()
                self.mask_changed.emit()
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._freehand_active:
            self._freehand_active = False
            points = self._polygon_points
            self._polygon_points = []
            self._draw_polygon_preview()
            if self.session is not None and len(points) >= 3:
                self.session.add_polygon(points)
                self.refresh_overlay()
                self.mask_changed.emit()
            else:
                self.message.emit("Freehand curve needs at least 3 points")
            event.accept()
            return
        if event.button() == Qt.LeftButton and self._painting and self.session is not None:
            self.finish_stroke()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    def __init__(self, image_dir: Path | None = None, mask_dir: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("CTA Lower-limb ROI Annotator")
        self.resize(1250, 820)
        self.session: AnnotationSession | None = None
        self._image_dir = image_dir
        self._mask_dir = mask_dir

        self.canvas = AnnotationCanvas()
        self.canvas.slice_requested.connect(self.navigate)
        self.canvas.mask_changed.connect(self.update_status)
        self.canvas.message.connect(lambda text: self.statusBar().showMessage(text, 3000))

        panel = QWidget()
        panel.setMaximumWidth(290)
        panel_layout = QVBoxLayout(panel)
        self.file_label = QLabel("No image folder selected")
        self.file_label.setWordWrap(True)
        self.progress_label = QLabel("0 / 0")
        self.state_label = QLabel("Unlabeled")
        panel_layout.addWidget(self.file_label)
        panel_layout.addWidget(self.progress_label)
        panel_layout.addWidget(self.state_label)

        panel_layout.addWidget(QLabel("Slices (click any file to jump)"))
        self.slice_list = QListWidget()
        self.slice_list.setMinimumHeight(190)
        self.slice_list.currentRowChanged.connect(self._slice_row_changed)
        self._syncing_slice_list = False
        panel_layout.addWidget(self.slice_list)

        open_images = QPushButton("Open Image Folder")
        open_masks = QPushButton("Select Mask Folder")
        open_images.clicked.connect(self.choose_image_dir)
        open_masks.clicked.connect(self.choose_mask_dir)
        panel_layout.addWidget(open_images)
        panel_layout.addWidget(open_masks)

        polygon = QPushButton("Polygon")
        circle = QPushButton("Circle")
        freehand = QPushButton("Freehand Curve")
        brush = QPushButton("Brush")
        eraser = QPushButton("Eraser")
        polygon.clicked.connect(lambda: self.select_tool("polygon"))
        circle.clicked.connect(lambda: self.select_tool("circle"))
        freehand.clicked.connect(lambda: self.select_tool("freehand"))
        brush.clicked.connect(lambda: self.select_tool("brush"))
        eraser.clicked.connect(lambda: self.select_tool("eraser"))
        panel_layout.addWidget(polygon)
        panel_layout.addWidget(circle)
        panel_layout.addWidget(freehand)
        panel_layout.addWidget(brush)
        panel_layout.addWidget(eraser)

        self.brush_size = QSpinBox()
        self.brush_size.setRange(1, 100)
        self.brush_size.setValue(10)
        self.brush_size.valueChanged.connect(self._set_brush_size)
        form = QFormLayout()
        form.addRow("Brush radius", self.brush_size)
        panel_layout.addLayout(form)

        opacity = QSlider(Qt.Horizontal)
        opacity.setRange(20, 220)
        opacity.setValue(105)
        opacity.valueChanged.connect(self._set_opacity)
        form.addRow("Overlay opacity", opacity)

        actions: list[tuple[str, object]] = [
            ("Clear (C)", self.clear),
            ("Copy Previous Mask (V)", self.copy_previous),
            ("Save (Ctrl+S)", self.save),
            ("Previous (A)", lambda: self.navigate(-1)),
            ("Next (D)", lambda: self.navigate(1)),
            ("Fit View (F)", self.fit_view),
        ]
        for label, callback in actions:
            button = QPushButton(label)
            button.clicked.connect(callback)  # type: ignore[arg-type]
            panel_layout.addWidget(button)
        panel_layout.addStretch(1)

        central = QWidget()
        main_layout = QHBoxLayout(central)
        main_layout.addWidget(self.canvas, 1)
        main_layout.addWidget(panel)
        self.setCentralWidget(central)
        shortcuts = QLabel(
            "A/D: previous/next | wheel: slices | Ctrl+wheel: zoom | middle drag: pan | "
            "Enter/double-click: close polygon | Ctrl+Z: undo | O: overlay"
        )
        shortcuts.setContentsMargins(8, 3, 8, 3)
        self.statusBar().addPermanentWidget(shortcuts, 1)
        self._install_shortcuts()

        if image_dir is not None and mask_dir is not None:
            self.load_session()

    def _install_shortcuts(self) -> None:
        bindings = {
            "A": lambda: self.navigate(-1),
            "D": lambda: self.navigate(1),
            "Ctrl+S": self.save,
            "Ctrl+Z": self.undo,
            "C": self.clear,
            "V": self.copy_previous,
            "O": self.toggle_overlay,
            "Return": self.canvas.close_polygon,
            "Enter": self.canvas.close_polygon,
            "Escape": self.canvas.cancel_polygon,
            "F": self.fit_view,
        }
        for key, callback in bindings.items():
            action = QAction(self)
            action.setShortcut(QKeySequence(key))
            action.setShortcutContext(Qt.WindowShortcut)
            action.triggered.connect(callback)  # type: ignore[arg-type]
            self.addAction(action)

    def _show_error(self, title: str, exc: Exception) -> None:
        QMessageBox.critical(self, title, str(exc))

    def choose_image_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select CTA image folder")
        if selected:
            self._image_dir = Path(selected)
            if self._mask_dir is None:
                self.choose_mask_dir()
            else:
                self.load_session()

    def choose_mask_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select or create mask folder")
        if selected:
            self._mask_dir = Path(selected)
            if self._image_dir is not None:
                self.load_session()

    def load_session(self) -> None:
        if self._image_dir is None or self._mask_dir is None:
            return
        try:
            if self.session is not None:
                self.session.close()
            self.session = AnnotationSession(self._image_dir, self._mask_dir)
            self.canvas.set_session(self.session)
            self._populate_slice_list()
            self.update_status()
        except Exception as exc:
            self._show_error("Cannot open folders", exc)

    def select_tool(self, mode: str) -> None:
        self.canvas.set_mode(mode)
        self.statusBar().showMessage(f"Tool: {mode}", 2000)

    def _set_brush_size(self, value: int) -> None:
        self.canvas.brush_radius = value

    def _set_opacity(self, value: int) -> None:
        self.canvas.overlay_opacity = value
        self.canvas.refresh_overlay()

    def update_status(self) -> None:
        if self.session is None:
            return
        self.file_label.setText(self.session.current_source.path.name)
        self.progress_label.setText(self.session.progress_text)
        if self.session.dirty:
            state = "Modified (not saved)"
        elif self.session.is_labeled:
            state = "Labeled / saved"
        else:
            state = "Unlabeled"
        self.state_label.setText(state)
        self._update_slice_list_state()

    def _slice_label(self, index: int) -> str:
        if self.session is None:
            return ""
        source = self.session.sources[index]
        if index == self.session.index and self.session.dirty:
            marker = "*"
        elif (self.session.mask_dir / source.path.name).is_file():
            marker = "✓"
        else:
            marker = "○"
        return f"{marker} {index + 1:04d}  {source.path.name}"

    def _populate_slice_list(self) -> None:
        if self.session is None:
            return
        self._syncing_slice_list = True
        self.slice_list.clear()
        for index in range(len(self.session.sources)):
            self.slice_list.addItem(self._slice_label(index))
        self.slice_list.setCurrentRow(self.session.index)
        self._syncing_slice_list = False

    def _update_slice_list_state(self) -> None:
        if self.session is None or self.slice_list.count() != len(self.session.sources):
            return
        self._syncing_slice_list = True
        for index in range(self.slice_list.count()):
            self.slice_list.item(index).setText(self._slice_label(index))
        self.slice_list.setCurrentRow(self.session.index)
        self.slice_list.scrollToItem(self.slice_list.item(self.session.index))
        self._syncing_slice_list = False

    def _slice_row_changed(self, row: int) -> None:
        if not self._syncing_slice_list and row >= 0:
            self.go_to_index(row)

    def go_to_index(self, index: int) -> None:
        if self.session is None:
            return
        try:
            self.canvas.finish_stroke()
            if self.session.navigate_to(index):
                self.canvas.cancel_polygon()
                self.canvas.refresh_image(fit=False)
            self.update_status()
        except Exception as exc:
            self._show_error("Cannot change slice", exc)
            self._update_slice_list_state()

    def navigate(self, delta: int) -> None:
        if self.session is None:
            return
        try:
            self.canvas.finish_stroke()
            if self.session.navigate(delta):
                self.canvas.cancel_polygon()
                self.canvas.refresh_image(fit=False)
                self.update_status()
        except Exception as exc:
            self._show_error("Cannot change slice", exc)

    def save(self) -> None:
        if self.session is None:
            return
        try:
            path = self.session.save()
            self.update_status()
            self.statusBar().showMessage(f"Saved: {path}", 3000)
        except Exception as exc:
            self._show_error("Save failed", exc)

    def undo(self) -> None:
        if self.session is not None and self.session.undo():
            self.canvas.refresh_overlay()
            self.update_status()

    def clear(self) -> None:
        if self.session is not None:
            self.session.clear()
            self.canvas.refresh_overlay()
            self.update_status()

    def copy_previous(self) -> None:
        if self.session is None:
            return
        try:
            self.session.copy_previous_mask()
            self.canvas.refresh_overlay()
            self.update_status()
        except Exception as exc:
            self._show_error("Copy previous mask failed", exc)

    def toggle_overlay(self) -> None:
        self.canvas.toggle_overlay()
        self.statusBar().showMessage(
            f"Overlay: {'on' if self.canvas.overlay_visible else 'off'}", 1500
        )

    def fit_view(self) -> None:
        if self.session is not None:
            self.canvas.resetTransform()
            self.canvas.fitInView(self.canvas.sceneRect(), Qt.KeepAspectRatio)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        try:
            if self.session is not None:
                self.session.close()
            event.accept()
        except Exception as exc:
            self._show_error("Save before exit failed", exc)
            event.ignore()


def run_app(image_dir: Path | None = None, mask_dir: Path | None = None) -> int:
    app = QApplication.instance() or QApplication([])
    window = MainWindow(image_dir=image_dir, mask_dir=mask_dir)
    window.show()
    return app.exec()
