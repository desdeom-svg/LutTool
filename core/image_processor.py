import cv2
import numpy as np
import math


class ImageProcessor:
    @staticmethod
    def apply_lut(image, lut):
        """应用 LUT 到图像 (支持 1D 和 3x1D)"""
        if image is None:
            return None
        
        # 如果是彩色图且 lut 是多通道的 [B_lut, G_lut, R_lut]
        if len(image.shape) == 3 and len(lut.shape) == 2 and lut.shape[0] == 3:
            # 分离通道，适配 3 通道 and 4 通道
            channels = cv2.split(image)
            if len(channels) >= 3:
                # 仅处理前三个颜色通道 (B, G, R)
                b_proc = cv2.LUT(channels[0], lut[0])
                g_proc = cv2.LUT(channels[1], lut[1])
                r_proc = cv2.LUT(channels[2], lut[2])
                
                new_channels = [b_proc, g_proc, r_proc]
                if len(channels) == 4:
                    # 保留 Alpha 通道不变
                    new_channels.append(channels[3])
                    
                return cv2.merge(new_channels)
            
        # 否则尝试通用 LUT (如果是彩色图且 lut 是单通道，Opencv 会自动对所有通道应用该 lut)
        return cv2.LUT(image, lut)

    @staticmethod
    def calculate_difference(original, processed):
        """计算差异图：abs(原图 - 处理图)。无增强，无归一化。"""
        if original is None or processed is None:
            return None
        # 确保尺寸一致
        if original.shape != processed.shape:
            processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
        
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
            
        best_score = -1
        best_gain = 1.0
        best_offset = 0
        best_lut = np.arange(256, dtype=np.uint8)
        
        gains = np.arange(0.5, 3.1, 0.25)
        offsets = range(-50, 51, 15)
        
        for gain in gains:
            for offset in offsets:
                lut = np.clip(np.arange(256) * gain + offset, 0, 255).astype(np.uint8)
                
                fg_res = cv2.LUT(fg_crop, lut)
                bg_res = cv2.LUT(bg_crop, lut)
                
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

        def calculate_single(o_ch, p_ch):
            if mode == "histogram":
                # --- 直方图模式 ---
                h_o, _ = np.histogram(o_ch, 256, [0, 256])
                h_p, _ = np.histogram(p_ch, 256, [0, 256])
                # 对直方图平滑处理减少噪声
                h_o = cv2.GaussianBlur(h_o.astype(np.float32), (7, 1), 0).flatten()
                h_p = cv2.GaussianBlur(h_p.astype(np.float32), (7, 1), 0).flatten()
                
                c_o = h_o.cumsum()
                c_p = h_p.cumsum()
                c_o_n = c_o / max(c_o[-1], 1)
                c_p_n = c_p / max(c_p[-1], 1)
                
                lut = np.zeros(256, dtype=np.uint8)
                for i in range(256):
                    target_val = c_o_n[i]
                    j = np.searchsorted(c_p_n, target_val)
                    lut[i] = np.clip(j, 0, 255).astype(np.uint8)
                
                # 强力平滑曲线
                lut = cv2.GaussianBlur(lut.astype(np.float32), (11, 1), 0).flatten()
                return np.clip(lut, 0, 255).astype(np.uint8)
            
            else:
                # --- 像素级映射模式 (Full/Sparse) ---
                o_flat = o_ch.flatten()
                p_flat = p_ch.flatten()
                p_sum = np.bincount(o_flat, weights=p_flat, minlength=256)
                p_cnt = np.bincount(o_flat, minlength=256)
                
                lut = np.arange(256, dtype=np.uint8)
                mask = p_cnt > 0
                lut[mask] = np.clip(p_sum[mask] / p_cnt[mask], 0, 255).astype(np.uint8)
                
                # 插值补全缺失
                if not np.all(mask):
                    ki = np.where(mask)[0]
                    mi = np.where(~mask)[0]
                    if ki.size > 0:
                        lut[mi] = np.interp(mi, ki, lut[ki]).astype(np.uint8)
                
                if mode == "sparse":
                    # 采样到 17 点再恢复
                    samples = np.linspace(0, 255, 17, dtype=int)
                    sparse_vals = lut[samples]
                    lut = np.interp(np.arange(256), samples, sparse_vals).astype(np.uint8)
                
                return np.clip(lut, 0, 255).astype(np.uint8)

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
