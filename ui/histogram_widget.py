from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QRect, QPointF
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QPolygonF
import cv2
import numpy as np

class HistogramWidget(QWidget):
    def __init__(self, title="直方图", parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.title = title
        self.hist_data = None 
        self.max_v = 255
        
    def set_image(self, cv_img):
        if cv_img is None:
            self.hist_data = None
            self.update()
            return
            
        # 自动探测位深 (Phase 29)
        max_v = 255
        if cv_img.dtype != np.uint8:
            max_val_found = np.max(cv_img)
            if max_val_found <= 1023: max_v = 1023
            elif max_val_found <= 4095: max_v = 4095
            else: max_v = 65535
        
        self.max_v = max_v
        # 统计精度上限 1024
        bins = min(max_v + 1, 1024)
        self.bins = bins

        if len(cv_img.shape) == 2:
            channels = [cv_img]
            colors = ['gray']
        else:
            channels = cv2.split(cv_img)
            colors = ['b', 'g', 'r']
            
        self.hist_data = {}
        self.peaks = {}
        for i, col in enumerate(colors):
            hist = cv2.calcHist([channels[i]], [0], None, [bins], [0, max_v + 1])
            if bins > 512:
                hist = cv2.GaussianBlur(hist, (7, 1), 0)
                
            peak_bin = np.argmax(hist)
            self.peaks[col] = int(peak_bin * (max_v + 1) / bins)
            
            cv2.normalize(hist, hist, 0, 100, cv2.NORM_MINMAX)
            self.hist_data[col] = hist.flatten()
            
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.contentsRect()
        margin_left = 40
        margin_bottom = 25
        margin_right = 10
        margin_top = 25
        
        canvas_w = rect.width() - margin_left - margin_right
        canvas_h = rect.height() - margin_top - margin_bottom
        
        painter.fillRect(rect, QColor(30, 30, 35))
        
        painter.setPen(QColor(80, 80, 85))
        painter.drawLine(margin_left, margin_top, margin_left, margin_top + canvas_h)
        painter.drawLine(margin_left, margin_top + canvas_h, margin_left + canvas_w, margin_top + canvas_h)

        painter.setPen(QColor(120, 120, 125))
        painter.setFont(self.font())
        for i, label in enumerate(["100%", "50%", "0%"]):
            y = margin_top + i * (canvas_h / 2)
            painter.drawText(5, int(y + 5), label)
            if i < 2:
                painter.setPen(QColor(50, 50, 55))
                painter.drawLine(margin_left, int(y), margin_left + canvas_w, int(y))
                painter.setPen(QColor(120, 120, 125))

        # X 轴
        max_v = self.max_v
        ticks = [0, int(max_v / 4), int(max_v / 2), int(max_v * 3 / 4), max_v]
            
        for t in ticks:
            x = margin_left + (t / max(1, max_v)) * canvas_w
            painter.drawText(int(x - 10), rect.height() - 5, str(t))
            painter.drawLine(int(x), margin_top + canvas_h, int(x), margin_top + canvas_h + 3)

        painter.setPen(QColor(200, 200, 205))
        painter.drawText(QRect(margin_left, 5, canvas_w, 20), Qt.AlignLeft, self.title)
        
        if self.hist_data is None:
            painter.drawText(QRect(margin_left, margin_top, canvas_w, canvas_h), Qt.AlignCenter, "等待图像导入...")
            return

        draw_colors = {
            'r': QColor(255, 80, 80, 150), 
            'g': QColor(80, 255, 80, 150), 
            'b': QColor(80, 80, 255, 150),
            'gray': QColor(220, 220, 220, 150)
        }
        
        for col, data in self.hist_data.items():
            painter.setPen(QPen(draw_colors[col].lighter(), 1.5))
            brush_color = QColor(draw_colors[col])
            brush_color.setAlpha(30)
            painter.setBrush(QBrush(brush_color))
            
            poly = QPolygonF()
            poly.append(QPointF(margin_left, margin_top + canvas_h))
            
            bin_count = len(data)
            for x_idx, val in enumerate(data):
                x = margin_left + (x_idx / max(1, bin_count - 1)) * canvas_w
                y = margin_top + canvas_h - (val / 100) * canvas_h
                poly.append(QPointF(x, y))
                
            poly.append(QPointF(margin_left + canvas_w, margin_top + canvas_h))
            painter.drawPolygon(poly)

            peak_val_real = self.peaks.get(col, 0)
            px = margin_left + (peak_val_real / max(1, max_v)) * canvas_w
            if 0 <= px <= rect.width():
                painter.setPen(QPen(draw_colors[col].lighter(), 1, Qt.DashLine))
                painter.drawLine(int(px), margin_top, int(px), margin_top + canvas_h)
                painter.setPen(QPen(draw_colors[col].lighter()))
                painter.drawText(int(px - 15), margin_top - 5, f"P:{peak_val_real}")
