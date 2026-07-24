from PIL import Image
import os

png_path = r"d:\Projects\pythonProject\LutTool\app_icon.png"
ico_path = r"d:\Projects\pythonProject\LutTool\app_icon.ico"

if os.path.exists(png_path):
    img = Image.open(png_path)
    # 调整大小以符合标准 ICO 要求 (256x256 max)
    icon_sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(ico_path, sizes=icon_sizes)
    print(f"Icon converted: {ico_path}")
else:
    print("Source PNG not found.")
