# LutTool Professional - 工业视觉 LUT 调优与反向推导工具

`LutTool Professional` 是一款专为工业视觉检测、机器视觉成像增强及相机图像 ISP 调试设计的专业级 **LUT (Look-Up Table, 查找表)** 可视化编辑、量化评估与反向工程拟合工具。系统支持多通道独立曲线微调、ROI 区域对比信号噪比（CNR）量化、PCHIP 单调立方样条插值，以及创新的“输入-输出图像反向 LUT 逆向推导”功能。

---

## 🚀 核心特性

- **多通道色彩与灰阶精准控制**：
  - 支持 RGB 全通道同步亮度/对比度调节，以及 R、G、B 分通道独立非线性映射；
  - 智能感知单通道灰度图（Grayscale）与彩色图（RGB/RGBA），自动锁定非法操作通道。
- **工业级 ROI 视觉质量量化评估**：
  - **CNR (Contrast-to-Noise Ratio)**：支持在图像中实时框选“缺陷区域”与“背景参考区”，量化计算信噪比增强增益；
  - **类间方差（Otsu Variance）**：计算前景缺陷与背景底纹的可分割度；
  - **图像信息熵（Information Entropy）**：衡量增强后细节保留与信息损失程度。
- **反向 LUT 逆向工程推导（Reverse LUT Extraction）**：
  - **直方图累积分布模式 (Histogram)**：基于色彩概率分布映射，免除空间位移与轻微畸变对齐误差；
  - **全灰阶精准模式 (Full Mapping)**：1:1 像素级映射，精准还原最细微的亮度级跳变；
  - **稀疏控制点模式 (Sparse PCHIP)**：自动拟合 17 个关键锚点，生成平滑解析曲线，便于直接嵌入 C++ 驱动或相机寄存器。
- **智能自动化算法箱**：
  - 自动直方图拉伸（Auto Stretch）、直方图均衡化（HE）、自适应增益/偏置智能寻优。

---

## 📁 目录结构

```text
LutTool/
├── core/                     # 核心算法引擎
│   ├── lut_engine.py         # LUT 曲线插值 (PCHIP)、通道映射与格式转换
│   ├── metrics.py            # CNR、类间方差、信息熵工业指标计算
│   └── reverse_lut.py        # 基于直方图与配对像素的反向 LUT 逆向算法
├── ui/                       # 图形界面组件 (PyQt / Tkinter)
│   ├── canvas.py             # 支持缩放、平移与双 ROI 框选的高性能画布
│   └── curve_editor.py       # 交互式三次样条曲线编辑器
├── app_icon.ico              # 应用程序图标
├── main.py                   # 启动主入口
├── run_lut_tool.bat          # Windows 双击快速运行脚本
└── README.md                 # 项目详细技术说明
```

---

## 📖 快速上手与使用

### 1. 运行环境配置
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy opencv-python pillow scipy matplotlib
```

### 2. 启动程序
```powershell
python main.py
```
或直接双击根目录下的 `run_lut_tool.bat`。

### 3. 基础工作流程
1. **加载图像**：点击左侧工具栏“导入图片”，选择待增强的原始工控图像；
2. **量化对比评测**：在图像上拖动框选第一块 ROI（红色，缺陷区），再框选第二块 ROI（绿色，背景区），系统自动显示当前 CNR 值；
3. **交互调光**：在右侧曲线面板中左键拖动控制点，右键新增/删除控制点，图像实时渲染更新。

### 4. 反向推导获取机台增强算法
1. 导入未经处理的机台原始图 (Input)；
2. 点击菜单 **“智能工具箱” -> “反向推导 LUT”**；
3. 选择生产端已增强的样张 (Output)，系统自动推导等效 1D/3D LUT 曲线并支持导出为 `.csv` 或工业标准格式。

### 5. 生产打包
```powershell
pyinstaller --noconsole --onefile --icon=app_icon.ico --name "LutTool_V2.0" main.py
```

---

## 🔒 资源与数据规范
- 实际现场采图（`data/`, `images/`, `lutData/`）与导出临时大表（`*.cube`）已通过 `.gitignore` 排除，确保 Git 仓库仅承载核心算法实现与轻量 UI 源码。
