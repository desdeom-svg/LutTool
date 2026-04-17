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
        self.hist_data = None # {'r': [], 'g': [], 'b': []}
        
    def set_image(self, cv_img):
        if cv_img is None:
            self.hist_data = None
            self.update()
            return
            
        # 计算直方图 (Phase 11: 增强兼容性)
        if len(cv_img.shape) == 2:
            # 灰度图
            channels = [cv_img]
            colors = ['gray']
        else:
            # 彩色图
            channels = cv2.split(cv_img)
            colors = ['b', 'g', 'r']
            
        self.hist_data = {}
        self.peaks = {} # Phase 13: 记录峰值位置
        for i, col in enumerate(colors):
            hist = cv2.calcHist([channels[i]], [0], None, [256], [0, 256])
            peak_idx = np.argmax(hist)
            self.peaks[col] = peak_idx
            cv2.normalize(hist, hist, 0, 100, cv2.NORM_MINMAX)
            self.hist_data[col] = hist.flatten()
            
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        rect = self.contentsRect()
        margin_left = 35
        margin_bottom = 20
        margin_right = 10
        margin_top = 25
        
        canvas_w = rect.width() - margin_left - margin_right
        canvas_h = rect.height() - margin_top - margin_bottom
        
        # 背景
        painter.fillRect(rect, QColor(30, 30, 35))
        
        # 绘制坐标轴 (Phase 12)
        painter.setPen(QColor(80, 80, 85))
        # Y轴
        painter.drawLine(margin_left, margin_top, margin_left, margin_top + canvas_h)
        # X轴
        painter.drawLine(margin_left, margin_top + canvas_h, margin_left + canvas_w, margin_top + canvas_h)

        # 绘制 Y 轴标签
        painter.setPen(QColor(120, 120, 125))
        painter.setFont(self.font())
        for i, label in enumerate(["100%", "50%", "0%"]):
            y = margin_top + i * (canvas_h / 2)
            painter.drawText(5, int(y + 5), label)
            # 绘制水平参考线
            if i < 2:
                painter.setPen(QColor(50, 50, 55))
                painter.drawLine(margin_left, int(y), margin_left + canvas_w, int(y))
                painter.setPen(QColor(120, 120, 125))

        # 绘制 X 轴标签 (密集刻度: 0, 64, 128, 192, 255)
        ticks = [0, 64, 128, 192, 255]
        for t in ticks:
            x = margin_left + (t / 255) * canvas_w
            painter.drawText(int(x - 10), rect.height() - 5, str(t))
            # 刻度短线
            painter.drawLine(int(x), margin_top + canvas_h, int(x), margin_top + canvas_h + 3)

        # 标题
        painter.setPen(QColor(200, 200, 205))
        painter.drawText(QRect(margin_left, 5, canvas_w, 20), Qt.AlignLeft, self.title)
        
        if self.hist_data is None:
            painter.drawText(QRect(margin_left, margin_top, canvas_w, canvas_h), Qt.AlignCenter, "等待图像导入...")
            return

        # 绘制曲线
        draw_colors = {
            'r': QColor(255, 80, 80, 180), 
            'g': QColor(80, 255, 80, 180), 
            'b': QColor(80, 80, 255, 180),
            'gray': QColor(220, 220, 220, 180)
        }
        
        for col, data in self.hist_data.items():
            painter.setPen(QPen(draw_colors[col].lighter(), 1.5))
            brush_color = QColor(draw_colors[col])
            brush_color.setAlpha(35)
            painter.setBrush(QBrush(brush_color))
            
            poly = QPolygonF()
            poly.append(QPointF(margin_left, margin_top + canvas_h))
            
            for x_idx, val in enumerate(data):
                x = margin_left + (x_idx / 255) * canvas_w
                y = margin_top + canvas_h - (val / 100) * canvas_h
                poly.append(QPointF(x, y))
                
            poly.append(QPointF(margin_left + canvas_w, margin_top + canvas_h))
            painter.drawPolygon(poly)

            # 绘制峰值标注 (Phase 13)
            peak_x_idx = self.peaks.get(col, 0)
            px = margin_left + (peak_x_idx / 255) * canvas_w
            
            painter.setPen(QPen(draw_colors[col].lighter(), 1, Qt.DashLine))
            painter.drawLine(int(px), margin_top, int(px), margin_top + canvas_h)
            
            # 在顶部标注值
            painter.setPen(QPen(draw_colors[col].lighter()))
            painter.drawText(int(px - 15), margin_top - 5, f"P:{peak_x_idx}")
