from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QSizePolicy
from PyQt5.QtCore import Qt, QPointF, pyqtSignal, QRect
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QBrush
import numpy as np

class CurveEditor(QFrame):
    curveChanged = pyqtSignal()

    def __init__(self, lut_manager, parent=None):
        super().__init__(parent)
        self.lut_manager = lut_manager
        self.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        self.setStyleSheet("background-color: #1e1e22; border: 1px solid #444;")
        
        # UI 结构
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.header = QLabel(" LUT 映射曲线 (输入 -> 输出)")
        self.header.setFixedHeight(25) # 显式限制高度，与图像 quadrant 保持一致
        self.header.setStyleSheet("background-color: #3d3d45; color: #e0e0e0; font-weight: bold; padding-left: 5px; border-bottom: 1px solid #444;")
        layout.addWidget(self.header)

        # 绘图区域
        self.canvas = QWidget()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.paintEvent = self._paint_canvas
        self.canvas.mousePressEvent = self._handle_press
        self.canvas.mouseMoveEvent = self._handle_move
        self.canvas.mouseReleaseEvent = self._handle_release
        self.canvas.wheelEvent = self._handle_wheel
        self.canvas.setMouseTracking(True)
        layout.addWidget(self.canvas)

        self.dragging_idx = -1
        self.margin = 40
        self.point_radius = 6
        self.zoom_scale = 1.0

    def _to_screen(self, x, y):
        # 考虑 zoom_scale (这里简单实现：只缩放绘图比例，不改变坐标轴)
        w = self.canvas.width() - 2 * self.margin
        h = self.canvas.height() - 2 * self.margin
        px = self.margin + (x / 255.0) * w
        py = self.canvas.height() - self.margin - (y / 255.0) * h
        return px, py

    def _from_screen(self, px, py):
        w = self.canvas.width() - 2 * self.margin
        h = self.canvas.height() - 2 * self.margin
        x = (px - self.margin) / w * 255.0
        y = (self.canvas.height() - self.margin - py) / h * 255.0
        return np.clip(x, 0, 255), np.clip(y, 0, 255)

    def _paint_canvas(self, event):
        painter = QPainter(self.canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.canvas.width()
        h = self.canvas.height()

        # 背景
        painter.fillRect(self.canvas.rect(), QColor(30, 30, 35))
        
        # 绘制坐标轴刻度与文字
        painter.setPen(QPen(QColor(150, 150, 160), 1))
        font = QFont("Segoe UI", 9)
        painter.setFont(font)

        for i in range(5):
            val = i * 64 if i < 4 else 255
            px, py = self._to_screen(val, val)
            
            # X 轴刻度文字
            painter.drawText(int(px) - 10, h - self.margin + 20, str(val))
            # Y 轴刻度文字
            painter.drawText(5, int(py) + 5, str(val))
            
            # 网格线
            painter.setPen(QPen(QColor(60, 60, 65), 1, Qt.DashLine))
            painter.drawLine(int(px), self.margin, int(px), h - self.margin)
            painter.drawLine(self.margin, int(py), w - self.margin, int(py))
            painter.setPen(QPen(QColor(150, 150, 160), 1))

        # 绘制主坐标轴
        painter.setPen(QPen(QColor(200, 200, 210), 2))
        painter.drawLine(self.margin, h - self.margin, w - self.margin, h - self.margin) # X
        painter.drawLine(self.margin, self.margin, self.margin, h - self.margin) # Y
        
        # 轴名称
        painter.drawText(w - self.margin - 40, h - self.margin - 10, "Input")
        painter.rotate(-90)
        painter.drawText(-self.margin - 60, self.margin + 20, "Output")
        painter.rotate(90)

        # 绘制原始参考曲线 (Phase 7: 红色虚线 y=x)
        ref_pen = QPen(QColor(255, 50, 50, 150), 1.0, Qt.DashLine)
        painter.setPen(ref_pen)
        p1_x, p1_y = self._to_screen(0, 0)
        p2_x, p2_y = self._to_screen(255, 255)
        painter.drawLine(QPointF(p1_x, p1_y), QPointF(p2_x, p2_y))

        # 绘制主 LUT 曲线
        lut = self.lut_manager.get_lut()
        
        # 根据频道设置颜色 (RGB: 蓝色偏白, R: 红, G: 绿, B: 蓝)
        ch_colors = {
            'RGB': QColor(100, 150, 255),
            'R': QColor(255, 80, 80),
            'G': QColor(80, 255, 80),
            'B': QColor(80, 80, 255)
        }
        curve_color = ch_colors.get(self.lut_manager.current_channel, QColor(100, 150, 255))
        
        painter.setPen(QPen(curve_color, 2))
        for i in range(255):
            x1, y1 = self._to_screen(i, lut[i])
            x2, y2 = self._to_screen(i + 1, lut[i+1])
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # 绘制控制点
        for i, (x, y) in enumerate(self.lut_manager.get_current_control_points()):
            px, py = self._to_screen(x, y)
            if i == self.dragging_idx:
                painter.setBrush(QBrush(QColor(255, 100, 100)))
            else:
                painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.drawEllipse(QPointF(px, py), self.point_radius, self.point_radius)

    def _handle_press(self, event):
        if event.button() == Qt.LeftButton:
            for i, (x, y) in enumerate(self.lut_manager.get_current_control_points()):
                px, py = self._to_screen(x, y)
                if (QPointF(px, py) - QPointF(event.x(), event.y())).manhattanLength() < 15:
                    self.dragging_idx = i
                    self.canvas.update()
                    return
            
            nx, ny = self._from_screen(event.x(), event.y())
            self.lut_manager.add_control_point(int(nx), int(ny))
            self.curveChanged.emit()
            self.canvas.update()
        
        elif event.button() == Qt.RightButton:
            for i, (x, y) in enumerate(self.lut_manager.get_current_control_points()):
                px, py = self._to_screen(x, y)
                if (QPointF(px, py) - QPointF(event.x(), event.y())).manhattanLength() < 15:
                    self.lut_manager.remove_control_point(i)
                    self.curveChanged.emit()
                    self.canvas.update()
                    break

    def _handle_move(self, event):
        if self.dragging_idx != -1:
            nx, ny = self._from_screen(event.x(), event.y())
            self.lut_manager.update_control_point(self.dragging_idx, int(nx), int(ny))
            self.curveChanged.emit()
            self.canvas.update()

    def _handle_release(self, event):
        self.dragging_idx = -1
        self.canvas.update()

    def _handle_wheel(self, event):
        # 曲线图的缩放这里简化为调整 margin (效果类似于放大核心区域)
        if event.angleDelta().y() > 0:
            self.margin = max(10, self.margin - 5)
        else:
            self.margin = min(100, self.margin + 5)
        self.canvas.update()
