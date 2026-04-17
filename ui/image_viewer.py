from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QFrame, QVBoxLayout, QLabel, QWidget
from PyQt5.QtCore import Qt, QPoint, QRectF, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QCursor, QColor, QPainter, QPen
import cv2
import numpy as np

class ImageViewer(QFrame):
    pixelClicked = pyqtSignal(int, int, list) # x, y, [r, g, b]
    viewChanged = pyqtSignal() # 用于同步视图

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.setStyleSheet("background-color: #2b2b2b; border: 1px solid #444;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        self.header = QLabel(f" {title}")
        self.header.setFixedHeight(25)
        self.header.setStyleSheet("background-color: #3d3d45; color: #e0e0e0; font-weight: bold; border: none;")
        layout.addWidget(self.header)
        
        # View
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setStyleSheet("border: none; background-color: transparent;")
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        layout.addWidget(self.view)
        
        # Status
        self.footer = QLabel(" 坐标: -, - | 像素: -")
        self.footer.setStyleSheet("color: #888; font-size: 11px; border: none; background-color: #2b2b2b;")
        layout.addWidget(self.footer)

        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.cv_img = None
        
        self.drawing_mode = None
        self.roi_defect = None
        self.roi_bg = None
        self.start_pos = None
        
        self.rect_item_defect = QGraphicsRectItem()
        self.rect_item_defect.setPen(QPen(QColor(255, 0, 0), 2))
        self.rect_item_defect.setZValue(10)
        self.scene.addItem(self.rect_item_defect)
        
        self.rect_item_bg = QGraphicsRectItem()
        self.rect_item_bg.setPen(QPen(QColor(0, 255, 0), 2))
        self.rect_item_bg.setZValue(10)
        self.scene.addItem(self.rect_item_bg)
        
        # 连接鼠标事件
        self.view.mousePressEvent = self._handle_mouse_press
        self.view.mouseMoveEvent = self._handle_mouse_move
        self.view.mouseReleaseEvent = self._handle_mouse_release
        self.view.wheelEvent = self._handle_wheel
        self.view.horizontalScrollBar().valueChanged.connect(lambda _: self.viewChanged.emit())
        self.view.verticalScrollBar().valueChanged.connect(lambda _: self.viewChanged.emit())

    def set_image(self, cv_img, keep_view=False):
        if cv_img is None:
            self.cv_img = None
            self.pixmap_item.setPixmap(QPixmap())
            return

        # 备份原始图像，确保它是 uint8 格式用于显示
        if cv_img.dtype != np.uint8:
            # 简单的归一化显示 (16-bit -> 8-bit)
            temp_img = (cv_img / (cv_img.max() / 255.0)).astype(np.uint8) if cv_img.max() > 0 else cv_img.astype(np.uint8)
        else:
            temp_img = cv_img

        self.cv_img = cv_img # 保留原始位深数据用于像素查询
        
        height, width = temp_img.shape[:2]
        bytes_per_line = temp_img.strides[0] # 自动获取真实的每行字节数，极其重要

        if len(temp_img.shape) == 3:
            channel = temp_img.shape[2]
            if channel == 3:
                # BGR -> RGB
                q_img = QImage(temp_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            elif channel == 4:
                # BGRA -> RGBA
                q_img = QImage(temp_img.data, width, height, bytes_per_line, QImage.Format_RGBA8888).rgbSwapped()
            else:
                # 其他情况降级为灰度
                gray = cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY)
                q_img = QImage(gray.data, width, height, width, QImage.Format_Grayscale8)
        else:
            q_img = QImage(temp_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_img.copy()) # copy() 确保内存安全
        self.pixmap_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        if not keep_view:
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def set_drawing_mode(self, mode):
        self.drawing_mode = mode
        if mode:
            self.view.setDragMode(QGraphicsView.NoDrag)
            self.view.setCursor(Qt.CrossCursor)
        else:
            self.view.setDragMode(QGraphicsView.ScrollHandDrag)
            self.view.unsetCursor()
            
    def clear_rois(self):
        self.roi_defect = None
        self.roi_bg = None
        self.rect_item_defect.setRect(0, 0, 0, 0)
        self.rect_item_bg.setRect(0, 0, 0, 0)

    def _handle_mouse_press(self, event):
        # 左键点击查询像素或框选
        if event.button() == Qt.LeftButton:
            scene_pos = self.view.mapToScene(event.pos())
            x, y = int(scene_pos.x()), int(scene_pos.y())
            
            if self.drawing_mode:
                self.start_pos = (x, y)
                return
                
            if self.cv_img is not None:
                h, w = self.cv_img.shape[:2]
                if 0 <= x < w and 0 <= y < h:
                    pixel = self.cv_img[y, x]
                    if len(pixel.shape) == 0: # 灰度图
                        pixel = [pixel, pixel, pixel]
                    else: # BGR
                        pixel = [pixel[2], pixel[1], pixel[0]] # 转为 RGB 列表
                    
                    self.footer.setText(f" 坐标: {x}, {y} | 像素: R:{pixel[0]} G:{pixel[1]} B:{pixel[2]}")
                    self.pixelClicked.emit(x, y, pixel)
        
        # 显式将事件分发给 QGraphicsView 的原始实现以处理拖拽
        QGraphicsView.mousePressEvent(self.view, event)

    def _handle_mouse_move(self, event):
        if self.drawing_mode and self.start_pos and (event.buttons() & Qt.LeftButton):
            scene_pos = self.view.mapToScene(event.pos())
            x, y = int(scene_pos.x()), int(scene_pos.y())
            w = x - self.start_pos[0]
            h = y - self.start_pos[1]
            
            rx = min(self.start_pos[0], x)
            ry = min(self.start_pos[1], y)
            rw = abs(w)
            rh = abs(h)
            
            if self.drawing_mode == 'defect':
                self.rect_item_defect.setRect(rx, ry, rw, rh)
            elif self.drawing_mode == 'bg':
                self.rect_item_bg.setRect(rx, ry, rw, rh)
            return
            
        QGraphicsView.mouseMoveEvent(self.view, event)

    def _handle_mouse_release(self, event):
        if self.drawing_mode and self.start_pos:
            scene_pos = self.view.mapToScene(event.pos())
            x, y = int(scene_pos.x()), int(scene_pos.y())
            w = x - self.start_pos[0]
            h = y - self.start_pos[1]
            
            rx = min(self.start_pos[0], x)
            ry = min(self.start_pos[1], y)
            rw = abs(w)
            rh = abs(h)
            
            if rw > 0 and rh > 0:
                if self.drawing_mode == 'defect':
                    self.roi_defect = [rx, ry, rw, rh]
                elif self.drawing_mode == 'bg':
                    self.roi_bg = [rx, ry, rw, rh]
            self.start_pos = None
            # Emit a special signal or we can just poll it from main window
            return
            
        QGraphicsView.mouseReleaseEvent(self.view, event)

    def _handle_wheel(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
            
        self.view.scale(zoom_factor, zoom_factor)
        self.viewChanged.emit()
