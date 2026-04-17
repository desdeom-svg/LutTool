import sys
import os

# 尝试修复 Anaconda 环境下的 PyQt5 DLL 加载问题
qt_path = os.path.join(os.path.dirname(sys.executable), 'Library', 'bin')
if os.path.exists(qt_path):
    os.environ['PATH'] = qt_path + os.pathsep + os.environ['PATH']
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(qt_path)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

def resource_path(relative_path):
    """ 获取资源的绝对路径，兼容 PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)
from ui.main_window import LutToolWindow

def main():
    app = QApplication(sys.argv)
    
    # 设置应用名称和图标 (如果有)
    app.setApplicationName("LutTool Professional")
    
    # 这里可以设置之前生成的图片作为图标
    # 使用相对路径并兼容打包模式
    icon_path = resource_path("app_icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    window = LutToolWindow()
    window.showMaximized()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
