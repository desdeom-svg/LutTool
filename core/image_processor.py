import cv2
import numpy as np
import math


class ImageProcessor:
    @staticmethod
    def apply_lut(image, lut, rois=None):
        """应用 LUT 到图像 (支持 1D 和 3x1D)"""
        if image is None:
            return None
        
        # 1. 计算处理后的图 (副本)
        processed = None
        
        # 针对 8位图且 LUT 长度适配时使用 OpenCV 加速
        if image.dtype == np.uint8 and lut.shape[-1] == 256:
            if len(image.shape) == 3 and len(lut.shape) == 2 and lut.shape[0] == 3:
                channels = cv2.split(image)
                if len(channels) >= 3:
                    b_proc = cv2.LUT(channels[0], lut[0])
                    g_proc = cv2.LUT(channels[1], lut[1])
                    r_proc = cv2.LUT(channels[2], lut[2])
                    new_channels = [b_proc, g_proc, r_proc]
                    if len(channels) == 4: new_channels.append(channels[3])
                    processed = cv2.merge(new_channels)
            
            if processed is None:
                processed = cv2.LUT(image, lut)
        else:
            # 针对高位深图或非 256 长度 LUT 使用 Numpy 映射
            # 安全裁剪，防止索引越界
            idx = np.clip(image, 0, lut.shape[-1] - 1)
            
            if len(image.shape) == 3 and len(lut.shape) == 2 and lut.shape[0] == 3:
                channels = cv2.split(idx)
                b_proc = lut[0][channels[0]]
                g_proc = lut[1][channels[1]]
                r_proc = lut[2][channels[2]]
                new_channels = [b_proc, g_proc, r_proc]
                if len(channels) == 4: new_channels.append(channels[3])
                processed = cv2.merge(new_channels)
            else:
                processed = lut[idx]
            
            # 强制保证输出类型与输入一致 (防止 uint16 图应用 uint8 LUT 后类型变动导致后续 absdiff 崩溃)
            processed = processed.astype(image.dtype)
            
        # 2. 如果没有 ROIs 或者是全局模式，直接返回 processed
        if not rois:
            return processed
            
        # 3. 如果有 ROIs，则进行局部替换
        result = image.copy()
        H, W = image.shape[:2]
        for x, y, w, h in rois:
            # 坐标安全转换与裁剪
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(W, int(x + w)), min(H, int(y + h))
            if x2 > x1 and y2 > y1:
                result[y1:y2, x1:x2] = processed[y1:y2, x1:x2]
                
        return result

    @staticmethod
    def calculate_difference(original, processed):
        """计算差异图：abs(原图 - 处理图)。无增强，无归一化。"""
        if original is None or processed is None:
            return None
        # 确保尺寸一致
        if original.shape[:2] != processed.shape[:2]:
            processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
        
        # 确保类型一致
        if original.dtype != processed.dtype:
            processed = processed.astype(original.dtype)
            
        diff = cv2.absdiff(original, processed)
        return diff

    @staticmethod
    def calculate_metrics(image, defect_rect=None, bg_rect=None):
        """计算核心量化指标：基于用户手动框选的缺陷与背景计算 CNR 与类间方差"""
        if image is None:
            return 0.0, 0.0
            
        if not defect_rect or not bg_rect:
            # 兼容：如果没有选框则返回 0
            return 0.0, 0.0
            
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                gray = image[:, :, 0] # 兜底逻辑
        else:
            gray = image
            
        dx, dy, dw, dh = defect_rect
        bx, by, bw, bh = bg_rect
        
        # 安全裁剪防止越界
        h, w = gray.shape[:2]
        dx, dy, dw, dh = max(0, dx), max(0, dy), min(w - dx, dw), min(h - dy, dh)
        bx, by, bw, bh = max(0, bx), max(0, by), min(w - bx, bw), min(h - by, bh)
        
        fg_pixels = gray[dy:dy+dh, dx:dx+dw]
        bg_pixels = gray[by:by+bh, bx:bx+bw]
        
        if fg_pixels.size == 0 or bg_pixels.size == 0:
            return 0.0, 0.0
            
        u_bg = np.mean(bg_pixels)
        u_fg = np.mean(fg_pixels)
        sigma_bg = np.std(bg_pixels)
        
        if sigma_bg == 0:  
            cnr = float(abs(u_fg - u_bg))
        else:
            cnr = abs(u_fg - u_bg) / sigma_bg
            
        # 类间方差
        w0 = fg_pixels.size / (fg_pixels.size + bg_pixels.size)
        w1 = bg_pixels.size / (fg_pixels.size + bg_pixels.size)
        
        var_between = w0 * w1 * ((u_fg - u_bg) ** 2)
        
        return round(float(cnr), 2), round(float(var_between), 2)

    @staticmethod
    def apply_gaussian_filter(image, ksize):
        if image is None: return None
        # ksize must be odd
        ksize = max(3, ksize)
        if ksize % 2 == 0: ksize += 1
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    @staticmethod
    def apply_median_filter(image, ksize):
        if image is None: return None
        ksize = max(3, ksize)
        if ksize % 2 == 0: ksize += 1
        return cv2.medianBlur(image, ksize)

    @staticmethod
    def calculate_entropy(image):
        if image is None: return 1.0 # 默认避免除0
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                gray = image[:, :, 0]
        else:
            gray = image
        
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum() # 分布比率
        
        entropy = 0
        for p in hist:
            if p > 0:
                entropy -= p[0] * math.log2(p[0])
                
        return max(entropy, 0.01) # 防止完全纯色时熵极小导致除0

    @staticmethod
    def search_optimal_lut(image, defect_rect=None, bg_rect=None):
        """自动寻优：搜索 Gain 和 Offset 让 Score = Contrast / (Std_bg * Entropy) 最大"""
        if image is None: return None
        if not defect_rect or not bg_rect:
            raise ValueError("缺少不良区域或背景区域框选数据，无法自动寻优。")
            
        # 安全裁剪
        # 灰度转换
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                gray = image[:, :, 0]
        else:
            gray = image
            
        h, w = gray.shape[:2]
        dx, dy, dw, dh = defect_rect
        bx, by, bw, bh = bg_rect
        dx, dy, dw, dh = max(0, dx), max(0, dy), min(w - dx, dw), min(h - dy, dh)
        bx, by, bw, bh = max(0, bx), max(0, by), min(w - bx, bw), min(h - by, bh)
        
        # 裁剪出两块区域的数据 (仅对区域进行拉伸测试更省资源)
        fg_crop = gray[dy:dy+dh, dx:dx+dw]
        bg_crop = gray[by:by+bh, bx:bx+bw]
        
        if fg_crop.size == 0 or bg_crop.size == 0:
            raise ValueError("框选区域超出边界或为空。")
            
        # 位深适配
        max_v = 255
        if gray.dtype != np.uint8:
            max_v = 4095 if np.max(gray) <= 4095 else 65535
            
        best_score = -1
        best_gain = 1.0
        best_offset = 0
        best_lut = np.arange(max_v + 1, dtype=gray.dtype)
        
        gains = np.arange(0.5, 3.1, 0.5)
        offsets = np.linspace(-max_v*0.2, max_v*0.2, 5)
        
        x_base = np.arange(max_v + 1)
        for gain in gains:
            for offset in offsets:
                lut = np.clip(x_base * gain + offset, 0, max_v).astype(gray.dtype)
                
                # 使用 numpy 映射代替 cv2.LUT 以兼容高位深
                fg_res = lut[fg_crop]
                bg_res = lut[bg_crop]
                
                entropy_fg = ImageProcessor.calculate_entropy(fg_res)
                entropy_bg = ImageProcessor.calculate_entropy(bg_res)
                # 使用整个感兴趣区域的平均熵作为特征 (或是简单取两块的均值)
                entropy = (entropy_fg + entropy_bg) / 2.0
                
                u_bg = np.mean(bg_res)
                u_fg = np.mean(fg_res)
                sigma_bg = np.std(bg_res)
                
                contrast = abs(u_fg - u_bg)
                sigma_bg = max(sigma_bg, 0.1)
                
                score = contrast / (sigma_bg * entropy)
                if score > best_score:
                    best_score = score
                    best_gain = gain
                    best_offset = offset
                    best_lut = lut
                    
        return {
            'gain': best_gain,
            'offset': best_offset,
            'score': best_score,
            'lut': best_lut
        }

    @staticmethod
    def derive_lut_from_images(original, processed, mode="histogram"):
        """
        基于图像对反向推导 LUT。
        mode: "sparse" (少量灰阶), "full" (全灰阶), "histogram" (直方图)
        """
        if original is None or processed is None:
            return None
        
        # 确保尺寸一致
        if original.shape[:2] != processed.shape[:2]:
            processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
            
        is_orig_color = len(original.shape) == 3 and original.shape[2] >= 3
        is_proc_color = len(processed.shape) == 3 and processed.shape[2] >= 3
        
        # 自动识别量程
        max_v = 255
        if original.dtype != np.uint8:
            max_val_found = np.max(original)
            if max_val_found <= 1023: max_v = 1023
            elif max_val_found <= 4095: max_v = 4095
            else: max_v = 65535
            
        lut_size = max_v + 1

        def calculate_single(o_ch, p_ch):
            dtype = original.dtype
            if mode == "histogram":
                # --- 直方图模式 ---
                h_o, _ = np.histogram(o_ch, lut_size, [0, lut_size])
                h_p, _ = np.histogram(p_ch, lut_size, [0, lut_size])
                h_o = cv2.GaussianBlur(h_o.astype(np.float32), (15, 1), 0).flatten()
                h_p = cv2.GaussianBlur(h_p.astype(np.float32), (15, 1), 0).flatten()
                
                c_o = h_o.cumsum()
                c_p = h_p.cumsum()
                c_o_n = c_o / max(c_o[-1], 1)
                c_p_n = c_p / max(c_p[-1], 1)
                
                lut = np.zeros(lut_size, dtype=dtype)
                for i in range(lut_size):
                    target_val = c_o_n[i]
                    j = np.searchsorted(c_p_n, target_val)
                    lut[i] = np.clip(j, 0, max_v).astype(dtype)
                return lut
            
            else:
                # --- 像素级映射模式 ---
                o_flat = o_ch.flatten()
                p_flat = p_ch.flatten()
                p_sum = np.bincount(o_flat, weights=p_flat, minlength=lut_size)
                p_cnt = np.bincount(o_flat, minlength=lut_size)
                
                lut = np.arange(lut_size, dtype=dtype)
                mask = p_cnt > 0
                lut[mask] = np.clip(p_sum[mask] / p_cnt[mask], 0, max_v).astype(dtype)
                
                if not np.all(mask):
                    ki = np.where(mask)[0]
                    mi = np.where(~mask)[0]
                    if ki.size > 0:
                        lut[mi] = np.interp(mi, ki, lut[ki]).astype(dtype)
                
                if mode == "sparse":
                    samples = np.linspace(0, max_v, 17, dtype=int)
                    sparse_vals = lut[samples]
                    lut = np.interp(np.arange(lut_size), samples, sparse_vals).astype(dtype)
                
                return lut

        if is_orig_color and is_proc_color:
            o_split = cv2.split(original)[:3]
            p_split = cv2.split(processed)[:3]
            return {
                'B': calculate_single(o_split[0], p_split[0]),
                'G': calculate_single(o_split[1], p_split[1]),
                'R': calculate_single(o_split[2], p_split[2])
            }
        else:
            o_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if is_orig_color else original
            p_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if is_proc_color else processed
            return {'RGB': calculate_single(o_gray, p_gray)}
