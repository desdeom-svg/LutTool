import sys
import cv2
import numpy as np
import os
from PyQt5.QtWidgets import (QMainWindow, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QTableWidget, QTableWidgetItem, QLabel, 
                             QHeaderView, QSplitter, QAction, QMenu, QMessageBox, QStyle, QGroupBox, QApplication)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QIcon, QFont

from core.lut_manager import LUTManager
from core.image_processor import ImageProcessor
from ui.curve_editor import CurveEditor
from ui.image_viewer import ImageViewer
from ui.histogram_widget import HistogramWidget

class LutToolWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LutTool Professional - 视觉化 LUT 编辑器")
        self.resize(1400, 900)
        
        self.lut_manager = LUTManager()
        self.image_processor = ImageProcessor()
        self.original_image = None
        self.filtered_image = None
        self.processed_image = None
        self.original_path = None # Phase 10: 记录原始路径
        
        # 采样模式状态 (Phase 8)
        self.picking_mode = None
        self.black_val = 0
        self.white_val = 255
        self.is_sync_view = False
        self.is_roi_mode = False
        
        self._init_ui()
        self._apply_styles()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧 2x2 严格布局
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(10)
        
        # 显式设置等比例权重 (Phase 6: 增加直方图行)
        self.grid_layout.setColumnStretch(0, 1)
        self.grid_layout.setColumnStretch(1, 1)
        self.grid_layout.setRowStretch(0, 2)
        self.grid_layout.setRowStretch(1, 2)
        self.grid_layout.setRowStretch(2, 1) # 直方图行稍窄
        
        self.view_orig = ImageViewer("原始图像 (Original)")
        self.view_curve = CurveEditor(self.lut_manager)
        self.view_processed = ImageViewer("效果预览 (Processed)")
        self.view_diff = ImageViewer("差异分析 (Difference)")
        
        # 新增直方图对比
        self.hist_orig = HistogramWidget("原始直方图")
        self.hist_proc = HistogramWidget("处理后直方图")
        
        # 3x2 排版
        self.grid_layout.addWidget(self.view_orig, 0, 0)
        self.grid_layout.addWidget(self.view_curve, 0, 1)
        self.grid_layout.addWidget(self.view_processed, 1, 0)
        self.grid_layout.addWidget(self.view_diff, 1, 1)
        self.grid_layout.addWidget(self.hist_orig, 2, 0)
        self.grid_layout.addWidget(self.hist_proc, 2, 1)
        
        # 信号连接
        self.view_curve.curveChanged.connect(self._on_curve_changed)
        self.view_orig.pixelClicked.connect(self._on_pixel_clicked)
        self.view_orig.roiAdded.connect(self._update_processed_images) # ROI 变化时刷新

        self.view_orig.viewChanged.connect(lambda: self._sync_views(self.view_orig))
        self.view_processed.viewChanged.connect(lambda: self._sync_views(self.view_processed))
        self.view_diff.viewChanged.connect(lambda: self._sync_views(self.view_diff))

        self._init_toolbar()

        # 右侧操作面板
        sidebar = QWidget()
        sidebar.setFixedWidth(320)
        side_layout = QVBoxLayout(sidebar)

        # 核心量化指标评测
        self.metrics_group = QGroupBox("核心量化指标评测")
        metrics_layout = QVBoxLayout()
        self.label_cnr = QLabel("当前 CNR: -")
        self.label_otsu = QLabel("类间方差: -")
        self.label_hint = QLabel("提示: 拉伸后 CNR > 3~5 , 抓取较稳定")
        self.label_hint.setStyleSheet("color: #a0a0ff; font-size: 11px;")
        metrics_layout.addWidget(self.label_cnr)
        metrics_layout.addWidget(self.label_otsu)
        metrics_layout.addWidget(self.label_hint)
        self.metrics_group.setLayout(metrics_layout)
        side_layout.addWidget(self.metrics_group)

        # 表格编辑区
        label_table = QLabel("LUT 映射表 (支持多选批量修改)")
        label_table.setWordWrap(True)
        label_table.setAlignment(Qt.AlignCenter)
        side_layout.addWidget(label_table)
        
        self.table = QTableWidget(256, 2)
        self.table.setHorizontalHeaderLabels(["输入", "输出"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionMode(QTableWidget.ContiguousSelection)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_table_menu)
        
        for i in range(256):
            self.table.setItem(i, 0, QTableWidgetItem(str(i)))
            self.table.item(i, 0).setFlags(Qt.ItemIsEnabled)
            self.table.setItem(i, 1, QTableWidgetItem(str(i)))
            
        self.table.itemChanged.connect(self._on_table_changed)
        side_layout.addWidget(self.table)
        
        # 底部按钮
        btn_box = QHBoxLayout()
        btn_load_lut = QPushButton("加载 LUT")
        btn_load_lut.clicked.connect(self._import_lut)
        btn_export_lut = QPushButton("导出 LUT")
        btn_export_lut.clicked.connect(self._export_lut)
        btn_box.addWidget(btn_load_lut)
        btn_box.addWidget(btn_export_lut)
        side_layout.addLayout(btn_box)

        main_layout.addWidget(self.grid_widget, 1)
        main_layout.addWidget(sidebar)

    def _init_toolbar(self):
        """创建顶部工具栏 (Phase 9)"""
        self.toolbar = self.addToolBar("常用工具")
        self.toolbar.setMovable(False)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toolbar.setIconSize(self.toolbar.iconSize() * 1.2) # 稍微加大图标

        # --- 图像管理 ---
        act_import = QAction(" 导入图片", self)
        act_import.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        act_import.triggered.connect(self._import_image)
        self.toolbar.addAction(act_import)

        act_export = QAction(" 导出效果图", self)
        act_export.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        act_export.triggered.connect(self._export_image)
        self.toolbar.addAction(act_export)

        self.toolbar.addSeparator()

        # --- 自动采样管理 (状态 Action) ---
        self.act_pick_black = QAction(" 采样黑场 (背景)", self)
        self.act_pick_black.setCheckable(True)
        self.act_pick_black.triggered.connect(lambda checked: self._start_picking('black', checked))
        self.toolbar.addAction(self.act_pick_black)

        self.act_pick_white = QAction(" 采样白场 (目标)", self)
        self.act_pick_white.setCheckable(True)
        self.act_pick_white.triggered.connect(lambda checked: self._start_picking('white', checked))
        self.toolbar.addAction(self.act_pick_white)
        
        self.toolbar.addSeparator()
        self.act_roi_defect = QAction(" 框选不良区域", self)
        self.act_roi_defect.setCheckable(True)
        self.act_roi_defect.triggered.connect(lambda checked: self._set_roi_mode('defect', checked))
        self.toolbar.addAction(self.act_roi_defect)
        
        self.act_roi_bg = QAction(" 框选背景区域", self)
        self.act_roi_bg.setCheckable(True)
        self.act_roi_bg.triggered.connect(lambda checked: self._set_roi_mode('bg', checked))
        self.toolbar.addAction(self.act_roi_bg)

        # --- ROI 局部模式 (新需求) ---
        self.toolbar.addSeparator()
        self.act_draw_roi = QAction(" 绘制 ROI", self)
        self.act_draw_roi.setCheckable(True)
        self.act_draw_roi.triggered.connect(lambda checked: self._set_roi_mode('roi', checked))
        self.toolbar.addAction(self.act_draw_roi)

        self.act_roi_mode = QAction(" ROI 模式", self)
        self.act_roi_mode.setCheckable(True)
        self.act_roi_mode.setChecked(False)
        self.act_roi_mode.setToolTip("开启时 LUT 仅对 ROI 区域生效，关闭时全局生效")
        self.act_roi_mode.triggered.connect(self._toggle_roi_apply_mode)
        self.toolbar.addAction(self.act_roi_mode)

        act_clear_roi = QAction(" 清除 ROI", self)
        act_clear_roi.setIcon(self.style().standardIcon(QStyle.SP_DialogDiscardButton))
        act_clear_roi.triggered.connect(lambda: [self.view_orig.clear_rois(), self._update_processed_images()])
        self.toolbar.addAction(act_clear_roi)
        
        # --- 曲线模式切换 (Phase 19) ---
        self.toolbar.addSeparator()
        from PyQt5.QtWidgets import QActionGroup
        mode_group = QActionGroup(self)
        
        self.act_mode_linear = QAction(" 折线模式", self)
        self.act_mode_linear.setCheckable(True)
        self.act_mode_linear.setChecked(False)
        self.act_mode_linear.setIcon(self.style().standardIcon(QStyle.SP_TitleBarNormalButton))
        self.act_mode_linear.triggered.connect(lambda: self._set_curve_mode('linear'))
        
        self.act_mode_smooth = QAction(" 平滑曲线", self)
        self.act_mode_smooth.setCheckable(True)
        self.act_mode_smooth.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        self.act_mode_smooth.setChecked(True)
        self.act_mode_smooth.triggered.connect(lambda: self._set_curve_mode('smooth'))
        
        mode_group.addAction(self.act_mode_linear)
        mode_group.addAction(self.act_mode_smooth)
        self.toolbar.addAction(self.act_mode_linear)
        self.toolbar.addAction(self.act_mode_smooth)
        
        # --- 滤波去噪 ---
        self.toolbar.addSeparator()
        from PyQt5.QtWidgets import QToolButton, QMenu
        filter_tool = QToolButton(self)
        filter_tool.setText(" 图片预处理滤波")
        filter_tool.setPopupMode(QToolButton.InstantPopup)
        filter_tool.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView))
        filter_tool.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        
        filter_menu = QMenu(self)
        act_gaussian = QAction("高斯滤波 (Gaussian Blur)", self)
        act_gaussian.triggered.connect(lambda: self._apply_filter("gaussian"))
        act_median = QAction("中值滤波 (Median Blur)", self)
        act_median.triggered.connect(lambda: self._apply_filter("median"))
        
        filter_menu.addAction(act_gaussian)
        filter_menu.addAction(act_median)
        filter_tool.setMenu(filter_menu)
        self.toolbar.addWidget(filter_tool)

        # --- 自动化增强工具 (Phase 16) ---
        self.toolbar.addSeparator()
        auto_tool = QToolButton(self)
        auto_tool.setText(" 智能工具箱") # 更名以包含更多工具
        auto_tool.setPopupMode(QToolButton.InstantPopup)
        auto_tool.setIcon(self.style().standardIcon(QStyle.SP_CommandLink))
        auto_tool.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        
        auto_menu = QMenu(self)
        act_auto_linear = QAction("自动百分比拉伸 (针对灰雾背景)", self)
        act_auto_linear.triggered.connect(self._run_auto_linear)
        act_auto_he = QAction("全局直方图均衡 (提升低对比度)", self)
        act_auto_he.triggered.connect(self._run_auto_he)
        
        act_auto_lut = QAction("自动对比度寻优 (Gain/Offset 最优分数)", self)
        act_auto_lut.triggered.connect(self._run_auto_optimal_lut)
        
        # 新增反向推导
        act_derive_lut = QAction("反向推导 LUT (基于 Input/Output 图片对)", self)
        act_derive_lut.setIcon(self.style().standardIcon(QStyle.SP_ArrowBack))
        act_derive_lut.triggered.connect(self._run_derive_lut)
        
        auto_menu.addAction(act_auto_linear)
        auto_menu.addAction(act_auto_he)
        auto_menu.addAction(act_auto_lut)
        auto_menu.addSeparator()
        auto_menu.addAction(act_derive_lut)
        auto_tool.setMenu(auto_menu)
        self.toolbar.addWidget(auto_tool)

        self.toolbar.addSeparator()

        # --- 全局控制 ---
        act_reset = QAction(" 彻底重置 ", self)
        act_reset.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        act_reset.triggered.connect(self._reset_curve)
        self.toolbar.addAction(act_reset)

        self.act_sync_view = QAction(" 开启图片跟随", self)
        self.act_sync_view.setCheckable(True)
        self.act_sync_view.triggered.connect(self._toggle_sync_view)
        self.toolbar.addAction(self.act_sync_view)

        self.toolbar.addSeparator()

        # --- 频道切换器 (Phase 21: 彩色图单通道调节) ---
        from PyQt5.QtWidgets import QComboBox
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["全通道 (RGB)", "红色通道 (R)", "绿色通道 (G)", "蓝色通道 (B)"])
        self.channel_combo.currentIndexChanged.connect(self._on_channel_selection_changed)
        self.toolbar.addWidget(QLabel(" 调节频道："))
        self.toolbar.addWidget(self.channel_combo)
        
        self.toolbar.addSeparator()
        
        # --- 帮助 ---
        act_help = QAction(" 使用帮助", self)
        act_help.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxInformation) )
        act_help.triggered.connect(self._show_help)
        self.toolbar.addAction(act_help)

    def _reset_curve(self):
        self.lut_manager.reset()
        if self.original_image is not None:
            self.filtered_image = self.original_image.copy()
            self.view_orig.clear_rois()
            self._set_roi_mode(None)
            self.view_orig.set_image(self.filtered_image)
            self.hist_orig.set_image(self.filtered_image)
            for v in [self.view_orig, self.view_processed, self.view_diff]:
                if v.scene.sceneRect().width() > 0:
                    v.view.fitInView(v.scene.sceneRect(), Qt.KeepAspectRatio)
                    
        self._on_curve_changed()
        self.view_curve.canvas.update()

    def _set_roi_mode(self, mode, checked=True):
        # 确定最终模式：如果取消选中，则模式为 None
        final_mode = mode if checked else None
        
        # 取消互斥按钮的选中状态
        self.act_roi_defect.setChecked(final_mode == 'defect')
        self.act_roi_bg.setChecked(final_mode == 'bg')
        self.act_draw_roi.setChecked(final_mode == 'roi')
        
        # 取消采样工具的选中
        self.act_pick_black.setChecked(False)
        self.act_pick_white.setChecked(False)
        
        # 更新图像查看器的状态
        self.view_orig.set_drawing_mode(final_mode)

    def _start_picking(self, ptype, checked=True):
        final_type = ptype if checked else None
        
        for act in [self.act_pick_black, self.act_pick_white]:
            act.setChecked(act.text().find('黑场') != -1 if final_type == 'black' else (act.text().find('白场') != -1 if final_type == 'white' else False))
            
        # 此时也要确保互斥的 ROI 按钮取消
        self.act_roi_defect.setChecked(False)
        self.act_roi_bg.setChecked(False)
        self.act_draw_roi.setChecked(False)
        
        if final_type:
            QApplication.setOverrideCursor(Qt.CrossCursor)
            self.view_orig.set_drawing_mode('pick') # 特殊模式用于采样
            self.picking_mode = final_type
        else:
            QApplication.restoreOverrideCursor()
            self.view_orig.set_drawing_mode(None)
            self.picking_mode = None

    def _toggle_roi_apply_mode(self, checked):
        """开启/关闭 ROI 局部应用模式"""
        self.is_roi_mode = checked
        if checked and not self.view_orig.generic_rois:
            QMessageBox.information(self, "信息", "ROI 模式已开启。请点击工具栏【绘制 ROI】在原图上圈选感兴趣区域。")
        self._update_processed_images()

    def _toggle_sync_view(self, checked):
        self.is_sync_view = checked

    def _sync_views(self, source_view, force=False):
        if not self.is_sync_view and not force: return
        
        self.view_orig.blockSignals(True)
        self.view_processed.blockSignals(True)
        self.view_diff.blockSignals(True)
        
        t = source_view.view.transform()
        h_val = source_view.view.horizontalScrollBar().value()
        v_val = source_view.view.verticalScrollBar().value()
        
        for v in [self.view_orig, self.view_processed, self.view_diff]:
            if v != source_view:
                v.view.setTransform(t)
                v.view.horizontalScrollBar().setValue(h_val)
                v.view.verticalScrollBar().setValue(v_val)
                
        self.view_orig.blockSignals(False)
        self.view_processed.blockSignals(False)
        self.view_diff.blockSignals(False)

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1a1a1e;
                color: #e0e0e0;
                font-family: 'Segoe UI', sans-serif;
            }
            QPushButton {
                background-color: #35353a;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 10px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45454a; border-color: #555; }
            QPushButton:pressed { background-color: #25252a; }
            QPushButton:checked { background-color: #2e5a88; border-color: #4a90e2; }
            QToolBar QToolButton:checked {
                background-color: #2e5a88;
                border: 1px solid #4a90e2;
                border-radius: 4px;
            }
            QTableWidget {
                background-color: #212124;
                gridline-color: #333;
                border: 1px solid #444;
                selection-background-color: #2e5a88;
            }
            QHeaderView::section {
                background-color: #2e2e33;
                padding: 6px;
                border: 1px solid #444;
                font-weight: bold;
            }
            QLabel { font-size: 13px; margin-bottom: 5px; }
        """)

    def _show_table_menu(self, pos):
        selection = self.table.selectedRanges()
        if not selection: return
        
        menu = QMenu()
        batch_act = QAction("批量修改选中值为...", self)
        batch_act.triggered.connect(self._batch_update_table)
        menu.addAction(batch_act)
        menu.exec_(self.table.mapToGlobal(pos))

    def _batch_update_table(self):
        from PyQt5.QtWidgets import QInputDialog
        max_v = self.lut_manager.max_val
        val, ok = QInputDialog.getInt(self, "批量修改", f"输入新的映射值 (0-{max_v}):", 0, 0, max_v)
        if ok:
            self.table.blockSignals(True)
            for range_ in self.table.selectedRanges():
                for r in range(range_.topRow(), range_.bottomRow() + 1):
                    self.table.item(r, 1).setText(str(val))
                    # 同时更新 lut_manager
                    lut = self.lut_manager.get_lut()
                    lut[r] = val
            self.table.blockSignals(False)
            self._on_curve_changed()
            self.view_curve.canvas.update()

    def _import_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择原始图像", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if file_path:
            # Phase 10: 使用原始模式加载 (UNCHANGED)
            self.original_path = file_path
            self.original_image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if self.original_image is None:
                QMessageBox.critical(self, "错误", "无法读取该图像格式")
                return
            
            # --- 自动识别位深与频道 (Phase 22/25) ---
            is_color = len(self.original_image.shape) >= 3 and self.original_image.shape[2] >= 3
            self.channel_combo.setEnabled(is_color)
            
            # 自动解析位深
            if self.original_image.dtype == np.uint8:
                self.lut_manager.set_bit_depth(8)
            else:
                max_val = self.original_image.max()
                if max_val <= 1023:
                    self.lut_manager.set_bit_depth(10)
                elif max_val <= 4095:
                    self.lut_manager.set_bit_depth(12)
                else:
                    self.lut_manager.set_bit_depth(16)
                
            # 同步 UI 表格量程
            self._update_table_range()
                
            if not is_color:
                self.channel_combo.setCurrentIndex(0) 
                self.lut_manager.set_current_channel('RGB')
                
            self.filtered_image = self.original_image.copy()
            self.view_orig.clear_rois()
            self.view_orig.set_image(self.filtered_image)
            self.hist_orig.set_image(self.filtered_image)
            self._update_processed_images()
            
            # Phase 25: 加载后自动同步初始缩放/位置
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(100, lambda: self._sync_views(self.view_orig, force=True))

    def _export_image(self):
        if self.processed_image is None or self.original_path is None:
            QMessageBox.warning(self, "警告", "没有可导出的图像。")
            return
        
        # Phase 10: 默认使用原图后缀
        orig_ext = os.path.splitext(self.original_path)[1]
        default_name = f"processed_{os.path.basename(self.original_path)}"
        file_path, _ = QFileDialog.getSaveFileName(self, "保存处理后的图像", default_name, f"Original Format (*{orig_ext});;All Files (*)")
        
        if file_path:
            cv2.imencode(orig_ext, self.processed_image)[1].tofile(file_path)
            QMessageBox.information(self, "成功", f"图像已导出至：\n{file_path}")

    def _import_lut(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "导入 LUT 文件", "", "LUT Files (*.cube *.txt *.lut);;All Files (*)")
        if file_path:
            if self.lut_manager.import_lut(file_path):
                self._on_curve_changed()
                self.view_curve.canvas.update()

    def _export_lut(self):
        """导出 LUT 文件，支持量程选择 (Phase 24)"""
        if self.original_image is None:
            is_color = True 
        else:
            # 判断是否为彩色图
            is_color = len(self.original_image.shape) == 3 and self.original_image.shape[2] >= 3
            
        # --- 量程模式选择 ---
        from PyQt5.QtWidgets import QInputDialog
        options = ["归一化 (0.0 - 1.0) - 通用标准", "原始值 (0 - 255) - 工业 raw"]
        # 默认选择索引 1: 原始值 (Phase 24)
        item, ok = QInputDialog.getItem(self, "选择 LUT 量程", "导出数据格式：", options, 1, False)
        if not ok:
            return
        normalize_mode = (item == options[0])

        # 弹出保存对话框
        ext_filter = "CUBE Files (*.cube);;TXT Files (*.txt);;All Files (*)"
        file_path, _ = QFileDialog.getSaveFileName(self, "导出 LUT 文件", "custom.cube", ext_filter)
        
        if file_path:
            # 执行导出
            success = self.lut_manager.export_lut_adaptive(
                file_path, 
                is_color=is_color, 
                normalize=normalize_mode
            )
            if success:
                QMessageBox.information(self, "成功", f"LUT 已导出至：\n{file_path}\n模式: {'彩色' if is_color else '灰度'}, 格式: {'归一化' if normalize_mode else '原始值'}")
            else:
                QMessageBox.critical(self, "错误", "导出失败，请检查文件路径。")

    @pyqtSlot()
    def _on_curve_changed(self):
        # 同步表格
        self.table.blockSignals(True)
        lut = self.lut_manager.get_lut()
        row_count = self.table.rowCount()
        for i in range(row_count):
            if i < len(lut):
                self.table.item(i, 1).setText(str(lut[i]))
        self.table.blockSignals(False)
        self._update_processed_images()

    def _update_table_range(self):
        """根据当前 LUT 分辨率更新表格行数 (Phase 26)"""
        max_v = self.lut_manager.max_val
        # UI 性能优化：表格上限 4096 (12bit)，16bit 下仅显示前 4096 行
        display_rows = min(max_v, 4095) + 1
        
        self.table.blockSignals(True)
        self.table.setRowCount(display_rows)
        for i in range(display_rows):
            if not self.table.item(i, 0):
                self.table.setItem(i, 0, QTableWidgetItem(str(i)))
                self.table.item(i, 0).setFlags(Qt.ItemIsEnabled)
            if not self.table.item(i, 1):
                self.table.setItem(i, 1, QTableWidgetItem(""))
        self.table.blockSignals(False)
        self._on_curve_changed()

    def _on_table_changed(self, item):
        if item.column() == 1:
            try:
                row = item.row()
                val = int(item.text())
                val = np.clip(val, 0, self.lut_manager.max_val)
                lut = self.lut_manager.get_lut()
                lut[row] = val
                self._update_processed_images()
                self.view_curve.canvas.update()
            except ValueError: pass

    def _update_processed_images(self):
        if self.filtered_image is not None:
            # 彩色图使用聚合 LUT，灰度图使用当前选中 LUT
            if len(self.filtered_image.shape) == 3:
                applied_lut = self.lut_manager.get_multi_channel_lut()
            else:
                applied_lut = self.lut_manager.get_lut()
                
            # 根据是否开启 ROI 模式决定传入的区域
            active_rois = self.view_orig.generic_rois if self.is_roi_mode else None
                
            self.processed_image = self.image_processor.apply_lut(self.filtered_image, applied_lut, rois=active_rois)
            diff = self.image_processor.calculate_difference(self.filtered_image, self.processed_image)
            self.view_processed.set_image(self.processed_image, keep_view=True)
            self.view_diff.set_image(diff, keep_view=True)
            self.hist_proc.set_image(self.processed_image)
            
            roi_df = self.view_orig.roi_defect
            roi_bg = self.view_orig.roi_bg
            
            if roi_df and roi_bg:
                try:
                    cnr, otsu = self.image_processor.calculate_metrics(self.processed_image, defect_rect=roi_df, bg_rect=roi_bg)
                    self.label_cnr.setText(f"当前 CNR: {cnr}")
                    self.label_otsu.setText(f"类间方差: {otsu}")
                except Exception:
                    self.label_cnr.setText("当前 CNR: 错误")
                    self.label_otsu.setText("类间方差: 错误")
            else:
                self.label_cnr.setText("当前 CNR: 请框选不良及背景")
                self.label_otsu.setText("类间方差: -")


    def _on_pixel_clicked(self, x, y, rgb):
        """点击图像时的回调"""
        if not self.picking_mode: return
        gray = int(0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
        
        if self.picking_mode == 'black':
            self.black_val = gray
            self.act_pick_black.setText(f" 采样黑场: {gray}")
            self._finish_picking()
        elif self.picking_mode == 'white':
            self.white_val = gray
            self.act_pick_white.setText(f" 采样白场: {gray}")
            self._finish_picking()

    def _finish_picking(self):
        """结束采样并应用逻辑"""
        self._start_picking(None)
        # 自动拉伸
        self.lut_manager.auto_stretch(self.black_val, self.white_val)
        self._on_curve_changed()
        self.view_curve.canvas.update()

    def _run_auto_linear(self):
        """执行自动百分比线性拉伸 (Phase 16)"""
        if self.original_image is None:
            QMessageBox.warning(self, "警告", "请先导入原始图像。")
            return
        if self.lut_manager.apply_auto_min_max(self.original_image):
            self._on_curve_changed()
            self.view_curve.canvas.update()

    def _run_auto_he(self):
        """执行全图直方图均衡 (Phase 16)"""
        if self.original_image is None:
            QMessageBox.warning(self, "警告", "请先导入原始图像。")
            return
        if self.lut_manager.apply_auto_he(self.original_image):
            self._on_curve_changed()
            self.view_curve.canvas.update()

    def _apply_filter(self, filter_type):
        if self.original_image is None:
            QMessageBox.warning(self, "警告", "请先导入原始图像。")
            return
            
        from PyQt5.QtWidgets import QInputDialog
        ksize, ok = QInputDialog.getInt(self, "输入滤波核大小", "选择核尺寸 (必须为奇数，如 3, 5, 7):", 3, 3, 21, 2)
        if ok:
            if filter_type == "gaussian":
                self.filtered_image = self.image_processor.apply_gaussian_filter(self.original_image, ksize)
            elif filter_type == "median":
                self.filtered_image = self.image_processor.apply_median_filter(self.original_image, ksize)
                
            self._update_processed_images()

    def _run_auto_optimal_lut(self):
        """执行自动化搜索最优 Gain/Offset"""
        if self.filtered_image is None:
            QMessageBox.warning(self, "警告", "请先导入原始图像。")
            return
            
        roi_df = self.view_orig.roi_defect
        roi_bg = self.view_orig.roi_bg
        
        if not roi_df or not roi_bg:
            QMessageBox.warning(self, "警告", "请先在上方工具栏采用框选工具选定【不良区域】与【背景区域】，以指定目标计算评估得分。")
            return
            
        QMessageBox.information(self, "开始搜索", "系统将自动寻找使 Score = Contrast / (Std*Entropy) 最大化的 LUT。\n这可能需要几秒钟，点击确定以开始计算。")
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            res = self.image_processor.search_optimal_lut(self.filtered_image, defect_rect=roi_df, bg_rect=roi_bg)
            QApplication.restoreOverrideCursor()
            
            if res:
                # 更新自身 LUT 映射表以与 UI 曲线保持同步
                best_lut = res['lut']
                lut_array = self.lut_manager.get_lut()
                for i in range(256):
                    lut_array[i] = best_lut[i]
                    
                self._on_curve_changed()
                self.view_curve.canvas.update()
                
                msg = f"智能寻优完成!\n\n找到的最佳参数:\nGain (对比): {res['gain']}\nOffset (亮度): {res['offset']}\n最终最佳得分 (Score): {res['score']:.4f}"
                QMessageBox.information(self, "寻优结果", msg)
            else:
                QMessageBox.warning(self, "警告", "图像无法生成有效的双峰分割，寻优失败。")
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "错误", f"自动寻优出错: {str(e)}")

    def _run_derive_lut(self):
        """反向推导 LUT 功能 (支持多模式)"""
        if self.original_image is None:
            QMessageBox.warning(self, "警告", "请先在主界面点击【导入图片】加载原始参考图 (Input)。")
            return
            
        # 判断当前原始图是否为彩色 (Phase 24)
        is_color = len(self.original_image.shape) == 3 and self.original_image.shape[2] >= 3

        # --- 模式选择 (Phase 23) ---
        from PyQt5.QtWidgets import QInputDialog
        modes_map = {
            "少量灰阶 (更平滑曲线)": "sparse",
            "全灰阶 (更精准映射)": "full",
            "直方图 (无视对齐误差)": "histogram"
        }
        # 默认值逻辑: 彩色 -> 直方图(2), 灰度 -> 全灰阶(1)
        default_idx = 2 if is_color else 1
        item, ok = QInputDialog.getItem(self, "选择推导模式", "推导策略：", list(modes_map.keys()), default_idx, False)
        if not ok:
            return
        selected_mode = modes_map[item]

        file_path, _ = QFileDialog.getOpenFileName(self, "选择处理后的对比图 (Output)", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff)")
        if not file_path:
            return
            
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # 加载对比图
            proc_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if proc_img is None:
                raise ValueError("无法读取对比图。")
                
            # 执行推导
            lut_results = self.image_processor.derive_lut_from_images(self.original_image, proc_img, mode=selected_mode)
            QApplication.restoreOverrideCursor()
            
            if lut_results:
                self.lut_manager.apply_full_lut_dict(lut_results)
                self._on_curve_changed()
                self.view_curve.canvas.update()
                QMessageBox.information(self, "成功", f"已使用【{item}】模式成功反选映射关系。")
            else:
                QMessageBox.warning(self, "失败", "无法根据这两张图片建立有效的亮度映射关系。")
                
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "错误", f"反向推导失败: {str(e)}")

    def _show_help(self):
        """显示操作手册"""
        help_text = """
        <h3>LutTool Professional 操作手册</h3>
        <p><b>1. 基础工作流:</b> 导入图片 -> 点击上方工具栏【框选不良区域】和【框选背景区域】在原图画框 -> 进行滤波或执行“智能自动寻优”。</p>
        <p><b>2. 图像核心量化指标评测:</b><br/>
           &nbsp;&nbsp;<b>CNR (对比噪声比)</b>: 取决于您框选的目标和背景的亮度绝对差异，与背景方差的比率。值越大拉伸越稳定（推荐 CNR > 3~5）。<br/>
           &nbsp;&nbsp;<b>信息熵 (Entropy)</b>: 代表图像涵盖特征细节信息的总和。拉伸极强可能会使图像噪点增多或变成黑白导致熵变为0。<br/>
           &nbsp;&nbsp;<b>智能寻优 Score 评分机制</b>: 兼顾对比度和图像自然平滑保留，利用公式 “Score = CNR / Entropy” 通过自动化算法寻找最佳拉伸参数。
        </p>
        <p><b>3. 快速隔离平滑降噪:</b> 若拍摄噪点过多，可选择“图像预处理滤波”。滤波的涂抹效果不改变原图备份；在预处理后能让背景方差变小以提升后续捕捉稳定度。</p>
        <p><b>4. 工具箱便捷按钮:</b> 面板顶部的“彻底重置”会清空所有您刚刚画的框选区、滤波状态以及恢复图片的 1:1 放缩。</p>
        """
        QMessageBox.about(self, "使用帮助", help_text)

    def _set_curve_mode(self, mode):
        """设置平滑/线性插值模式 (Phase 19)"""
        self.lut_manager.interpolation_mode = mode
        self.lut_manager.update_all_luts() # 确保全局更新
        self._on_curve_changed()
        self.view_curve.canvas.update()

    def _on_channel_selection_changed(self, index):
        """处理频道切换"""
        channels = ['RGB', 'R', 'G', 'B']
        self.lut_manager.set_current_channel(channels[index])
        
        # 刷新 UI
        self._on_curve_changed() # 更新表格
        self.view_curve.canvas.update() # 更新绘图颜色和曲线

import sys
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = LutToolWindow()
    window.show()
    sys.exit(app.exec_())
