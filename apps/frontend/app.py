"""
AIVC Frontend - DCT Transform Coding Interface
DCT变换编码图像处理界面
"""

import gradio as gr
import numpy as np
import os
import time
import hashlib
from PIL import Image

# 全局状态
selected_bases = set(range(64))  # 默认选中所有64个DCT基
current_image = None  # 当前输入图像
current_quality = 50  # 当前质量因子
DEBUG_ENABLED = os.getenv("AIVC_DEBUG", "1").strip().lower() not in {"0", "false", "no", "off"}
BLOCK_SIZE = 8
BASE_LUMA_QUANT_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float32)
IMAGE_CACHE = {
    "image_signature": None,
    "img_rgb": None,
    "img_gray": None,
    "img_cbcr": None,
    "img_padded": None,
    "dct_blocks": None,
    "h": 0,
    "w": 0,
    "h_new": 0,
    "w_new": 0,
    "sample_pixels": None,
    "sample_dct": None,
}
RENDER_CACHE = {"key": None, "result": None}
RECON_CACHE = {"key": None, "output_img": None, "output_luma": None, "psnr_text": None}
GRID_CACHE = {"key": None, "grid": None}


def debug_log(message):
    """统一调试日志输出，便于排查卡顿位置。"""
    if DEBUG_ENABLED:
        now = time.strftime("%H:%M:%S")
        print(f"[AIVC][{now}] {message}", flush=True)


def generate_dct_bases(size=8):
    """生成8x8 DCT基函数图像"""
    bases = np.zeros((size * size, size, size))
    for u in range(size):
        for v in range(size):
            basis = np.zeros((size, size))
            for x in range(size):
                for y in range(size):
                    alpha_u = np.sqrt(1 / size) if u == 0 else np.sqrt(2 / size)
                    alpha_v = np.sqrt(1 / size) if v == 0 else np.sqrt(2 / size)
                    basis[x, y] = alpha_u * alpha_v * np.cos(
                        (2 * x + 1) * u * np.pi / (2 * size)
                    ) * np.cos((2 * y + 1) * v * np.pi / (2 * size))
            bases[u * size + v] = basis
    return bases


def create_dct_matrix(size=8):
    """生成DCT变换矩阵 C，使得 DCT = C @ block @ C.T"""
    matrix = np.zeros((size, size), dtype=np.float32)
    factor = np.pi / (2 * size)
    for u in range(size):
        alpha = np.sqrt(1 / size) if u == 0 else np.sqrt(2 / size)
        for x in range(size):
            matrix[u, x] = alpha * np.cos((2 * x + 1) * u * factor)
    return matrix


DCT_MATRIX = create_dct_matrix(BLOCK_SIZE)
DCT_MATRIX_T = DCT_MATRIX.T
DCT_BASES = generate_dct_bases(BLOCK_SIZE)


def get_image_signature(img_rgb):
    """根据RGB图生成稳定签名，用于缓存命中判断。"""
    digest = hashlib.blake2b(img_rgb.tobytes(), digest_size=8).hexdigest()
    h, w = img_rgb.shape[:2]
    return f"{h}x{w}:{digest}"


def normalize_input_image(input_image):
    """统一转换为 RGB / Y / CbCr 三个通道表示。"""
    if isinstance(input_image, np.ndarray):
        arr = np.asarray(input_image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            img_rgb = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3:
            if arr.shape[2] == 1:
                img_rgb = np.repeat(arr[:, :, :1], 3, axis=2)
            else:
                img_rgb = arr[:, :, :3]
        else:
            raise ValueError(f"不支持的图像维度: {arr.shape}")
    else:
        img_rgb = np.array(input_image.convert("RGB"), dtype=np.uint8)

    ycbcr = np.array(Image.fromarray(img_rgb, "RGB").convert("YCbCr"), dtype=np.uint8)
    img_gray = ycbcr[:, :, 0]
    img_cbcr = ycbcr[:, :, 1:]
    return img_rgb, img_gray, img_cbcr


def build_mask(selected_set):
    """根据选中的DCT基生成8x8掩码。"""
    mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)
    if not selected_set:
        return mask
    idx_array = np.fromiter(selected_set, dtype=np.int32)
    mask[idx_array // BLOCK_SIZE, idx_array % BLOCK_SIZE] = 1.0
    return mask


def rebuild_image_cache(img_rgb, img_gray, img_cbcr, image_signature):
    """图像变化时重建图像级缓存（整图DCT只做一次）。"""
    h, w = img_gray.shape
    h_pad = (BLOCK_SIZE - h % BLOCK_SIZE) % BLOCK_SIZE
    w_pad = (BLOCK_SIZE - w % BLOCK_SIZE) % BLOCK_SIZE
    img_padded = np.pad(img_gray, ((0, h_pad), (0, w_pad)), mode="edge")
    h_new, w_new = img_padded.shape
    rows = h_new // BLOCK_SIZE
    cols = w_new // BLOCK_SIZE

    blocks = img_padded.reshape(rows, BLOCK_SIZE, cols, BLOCK_SIZE).transpose(0, 2, 1, 3).astype(np.float32)
    blocks_shifted = blocks - 128.0
    dct_blocks = np.matmul(np.matmul(DCT_MATRIX, blocks_shifted), DCT_MATRIX_T)

    center_row = h // 2 // BLOCK_SIZE
    center_col = w // 2 // BLOCK_SIZE
    sample_y = center_row * BLOCK_SIZE
    sample_x = center_col * BLOCK_SIZE
    sample_pixels = img_padded[sample_y:sample_y + BLOCK_SIZE, sample_x:sample_x + BLOCK_SIZE]
    sample_dct = dct_blocks[center_row, center_col]

    IMAGE_CACHE.update({
        "image_signature": image_signature,
        "img_rgb": img_rgb.copy(),
        "img_gray": img_gray.copy(),
        "img_cbcr": img_cbcr.copy(),
        "img_padded": img_padded,
        "dct_blocks": dct_blocks,
        "h": h,
        "w": w,
        "h_new": h_new,
        "w_new": w_new,
        "sample_pixels": sample_pixels,
        "sample_dct": sample_dct,
    })

    # 图像变化时，清理依赖旧图像的缓存
    RENDER_CACHE["key"] = None
    RENDER_CACHE["result"] = None
    RECON_CACHE["key"] = None
    RECON_CACHE["output_img"] = None
    RECON_CACHE["output_luma"] = None
    RECON_CACHE["psnr_text"] = None


def ensure_image_cache(img_rgb, img_gray, img_cbcr):
    """确保图像级缓存可用，返回(cache_hit, image_signature)。"""
    image_signature = get_image_signature(img_rgb)
    if IMAGE_CACHE["image_signature"] == image_signature:
        return True, image_signature
    rebuild_image_cache(img_rgb, img_gray, img_cbcr, image_signature)
    return False, image_signature


def reconstruct_output_img(mask, quality_factor):
    """使用缓存的整图DCT系数，执行掩码+量化+IDCT重建。"""
    quant_table = get_adjusted_quant_table(quality_factor)
    masked_dct = IMAGE_CACHE["dct_blocks"] * mask
    quantized_dct = np.round(masked_dct / quant_table)
    dequantized_dct = quantized_dct * quant_table
    reconstructed_blocks = np.matmul(np.matmul(DCT_MATRIX_T, dequantized_dct), DCT_MATRIX) + 128.0
    reconstructed = reconstructed_blocks.transpose(0, 2, 1, 3).reshape(IMAGE_CACHE["h_new"], IMAGE_CACHE["w_new"])
    output_luma = np.clip(reconstructed, 0, 255).astype(np.uint8)[:IMAGE_CACHE["h"], :IMAGE_CACHE["w"]]

    output_ycbcr = np.empty((IMAGE_CACHE["h"], IMAGE_CACHE["w"], 3), dtype=np.uint8)
    output_ycbcr[:, :, 0] = output_luma
    output_ycbcr[:, :, 1:] = IMAGE_CACHE["img_cbcr"]
    output_rgb = np.array(Image.fromarray(output_ycbcr, "YCbCr").convert("RGB"), dtype=np.uint8)
    return output_luma, output_rgb


def create_dct_grid(selected_set, size=8):
    """创建DCT基函数选择网格图像"""
    grid_key = tuple(sorted(selected_set))
    if size == BLOCK_SIZE and GRID_CACHE["key"] == grid_key:
        return GRID_CACHE["grid"]

    bases = DCT_BASES if size == BLOCK_SIZE else generate_dct_bases(size)
    grid_size = size * size
    cell_size = 32
    
    # 创建网格图像
    grid_img = np.ones((size * cell_size, size * cell_size, 3), dtype=np.uint8) * 255
    
    for i in range(grid_size):
        u = i // size
        v = i % size
        
        # 归一化基函数到0-255
        basis = bases[i]
        basis_norm = ((basis - basis.min()) / (basis.max() - basis.min() + 1e-8) * 255).astype(np.uint8)
        
        # 缩放到cell_size
        basis_img = np.array(Image.fromarray(basis_norm).resize((cell_size - 4, cell_size - 4)))
        
        # 放置到网格中
        y_start = u * cell_size + 2
        x_start = v * cell_size + 2
        
        # 根据选中状态设置边框颜色
        if i in selected_set:
            # 选中：绿色边框
            grid_img[y_start - 2:y_start + cell_size - 2, x_start - 2:x_start + cell_size - 2] = [0, 200, 0]
        else:
            # 未选中：红色边框
            grid_img[y_start - 2:y_start + cell_size - 2, x_start - 2:x_start + cell_size - 2] = [200, 0, 0]
        
        grid_img[y_start:y_start + cell_size - 4, x_start:x_start + cell_size - 4] = np.stack([basis_img] * 3, axis=-1)
    
    if size == BLOCK_SIZE:
        GRID_CACHE["key"] = grid_key
        GRID_CACHE["grid"] = grid_img

    return grid_img


def dct_transform(block):
    """8x8 DCT变换"""
    return np.matmul(np.matmul(DCT_MATRIX, block), DCT_MATRIX_T)


def idct_transform(coeffs):
    """8x8 IDCT逆变换"""
    return np.matmul(np.matmul(DCT_MATRIX_T, coeffs), DCT_MATRIX)


def normalize_quality_factor(quality):
    """将质量因子规范到[1, 100]的整数。"""
    try:
        quality_int = int(round(float(quality)))
    except (TypeError, ValueError):
        return 50
    return min(100, max(1, quality_int))


def get_adjusted_quant_table(quality):
    """根据质量因子获取JPEG量化表。"""
    quality_int = normalize_quality_factor(quality)
    scale = 50 / quality_int if quality_int < 50 else 2 - quality_int / 50
    return np.clip(BASE_LUMA_QUANT_TABLE * scale, 1, 255)


def quantize(coeffs, quality=50):
    """JPEG量化"""
    adjusted_quant = get_adjusted_quant_table(quality)
    quantized = np.round(coeffs / adjusted_quant)
    return quantized, adjusted_quant


def calculate_psnr(original, reconstructed):
    """计算PSNR"""
    mse = np.mean((original.astype(float) - reconstructed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def process_image(input_image, selected_bases_str, quality_factor, click_info):
    """处理图像并生成所有输出"""
    global selected_bases
    start_time = time.perf_counter()
    quality_factor = normalize_quality_factor(quality_factor)
    debug_log(
        f"process_image 开始: quality={quality_factor}, click={click_info}, selected={len(selected_bases)}"
    )
    
    if input_image is None:
        debug_log("process_image 提前返回: 未上传图像")
        return [None] * 9 + ["请选择图像", f"已选择 {len(selected_bases)}/64 个DCT基"]
    
    # 处理点击事件(idx:基序号, u和v代表位置坐标.这两种是一个意思)
    selection_changed = False
    if click_info is not None:
        x, y = int(click_info[0]), int(click_info[1])
        cell_size = 32
        v = x // cell_size
        u = y // cell_size
        if 0 <= u < 8 and 0 <= v < 8:
            idx = u * 8 + v
            if idx in selected_bases:
                selected_bases.discard(idx)
                selection_changed = True
                debug_log(f"DCT基取消选择: idx={idx} (u={u}, v={v})")
            else:
                selected_bases.add(idx)
                selection_changed = True
                debug_log(f"DCT基已选择: idx={idx} (u={u}, v={v})")
        else:
            debug_log(f"DCT网格点击越界，忽略: x={x}, y={y}")
    
    # 统一转换到 RGB + Y/CbCr，DCT作用于Y通道，输出时合成回彩色
    img_rgb, img_gray, img_cbcr = normalize_input_image(input_image)
    h, w = img_gray.shape
    debug_log(f"输入图像尺寸: h={h}, w={w}, channels=3")

    # 图像缓存：仅在图像内容变化时重建整图DCT
    cache_start = time.perf_counter()
    image_cache_hit, image_signature = ensure_image_cache(img_rgb, img_gray, img_cbcr)
    cache_elapsed = (time.perf_counter() - cache_start) * 1000
    if image_cache_hit:
        debug_log(f"图像缓存命中: 耗时 {cache_elapsed:.1f} ms")
    else:
        h_new = IMAGE_CACHE["h_new"]
        w_new = IMAGE_CACHE["w_new"]
        h_pad = h_new - h
        w_pad = w_new - w
        total_blocks = (h_new // BLOCK_SIZE) * (w_new // BLOCK_SIZE)
        debug_log(
            f"图像缓存重建完成: h_pad={h_pad}, w_pad={w_pad}, padded=({h_new}, {w_new}), "
            f"total_blocks={total_blocks}, 耗时 {cache_elapsed:.1f} ms"
        )

    # 渲染缓存：完全同状态时，直接返回结果，避免无意义重算
    selected_key = tuple(sorted(selected_bases))
    render_key = (image_signature, selected_key, quality_factor)
    if RENDER_CACHE["key"] == render_key:
        debug_log("渲染缓存命中: 状态未变化，跳过重建")
        return RENDER_CACHE["result"]

    mask = build_mask(selected_bases)

    # 创建DCT基选择网格
    grid_start = time.perf_counter()
    dct_grid = create_dct_grid(selected_bases)
    debug_log(f"DCT网格生成完成: 耗时 {(time.perf_counter() - grid_start) * 1000:.1f} ms")

    # 重建缓存：图像、基选择和质量因子不变时复用输出图
    recon_key = (image_signature, selected_key, quality_factor)
    if RECON_CACHE["key"] == recon_key:
        output_img = RECON_CACHE["output_img"]
        output_luma = RECON_CACHE["output_luma"]
        psnr_text = RECON_CACHE["psnr_text"]
        debug_log(
            f"重建缓存命中: selection_changed={selection_changed}, 复用output_img, {psnr_text}"
        )
    else:
        total_rows = IMAGE_CACHE["h_new"] // BLOCK_SIZE
        total_cols = IMAGE_CACHE["w_new"] // BLOCK_SIZE
        total_blocks = total_rows * total_cols
        debug_log(
            f"开始块重建: total_blocks={total_blocks}, rows={total_rows}, cols={total_cols}, "
            "mode=cache_dct+mask+quantize+vector_idct"
        )
        recon_start = time.perf_counter()
        output_luma, output_img = reconstruct_output_img(mask, quality_factor)
        recon_elapsed = (time.perf_counter() - recon_start) * 1000

        # 计算PSNR(衡量“重建图像和原图有多接近”的指标,数值越大表示重建图像质量越好)
        psnr = calculate_psnr(IMAGE_CACHE["img_gray"], output_luma)
        psnr_text = f"PSNR: {psnr:.2f} dB"
        debug_log(f"块重建完成: 耗时 {recon_elapsed:.1f} ms, {psnr_text}")

        RECON_CACHE["key"] = recon_key
        RECON_CACHE["output_img"] = output_img
        RECON_CACHE["output_luma"] = output_luma
        RECON_CACHE["psnr_text"] = psnr_text

    # 获取中心区域的8x8块用于显示矩阵（从缓存读取）
    sample_pixels = IMAGE_CACHE["sample_pixels"]
    selected_dct_coeffs = IMAGE_CACHE["sample_dct"] * mask
    quantized, quant_table = quantize(selected_dct_coeffs, quality_factor)
    
    # 格式化矩阵显示
    def format_matrix(mat, title):
        lines = [f"{title}:"]
        for row in mat:
            lines.append("  " + "  ".join([f"{int(x):4d}" for x in row]))
        return "\n".join(lines)
    
    input_pixels = format_matrix(sample_pixels, "Input Y Pixel Values")
    dct_coeffs_str = format_matrix(selected_dct_coeffs, "Selected DCT Coefficients")
    quantized_str = format_matrix(quantized, "Quantised DCT Coefficients")
    
    # 反量化和逆变换
    dequantized = quantized * quant_table
    dequantized_str = format_matrix(dequantized, "Dequantised DCT Coefficients")
    
    inv_transform = idct_transform(dequantized)
    inv_transform_str = format_matrix(inv_transform, "Inverse Transform Coefficients")

    output_pixels = format_matrix(
        np.clip(inv_transform + 128.0, 0, 255).astype(np.uint8),
        "Output Y Pixel Values"
    )
    
    selected_info = f"已选择 {len(selected_bases)}/64 个DCT基"
    result = [
        IMAGE_CACHE["img_rgb"],  # 输入图像
        dct_grid,  # DCT基选择网格
        output_img,  # 输出图像
        input_pixels,  # 输入像素值
        dct_coeffs_str,  # DCT系数
        quantized_str,  # 量化后系数
        dequantized_str,  # 反量化系数
        inv_transform_str,  # 逆变换系数
        output_pixels,  # 输出像素值
        psnr_text,  # PSNR信息
        selected_info  # 选中信息
    ]
    RENDER_CACHE["key"] = render_key
    RENDER_CACHE["result"] = result
    total_elapsed_ms = (time.perf_counter() - start_time) * 1000
    debug_log(f"process_image 完成: 总耗时 {total_elapsed_ms:.1f} ms")
    return result


def select_all():
    """选择所有基"""
    global selected_bases
    selected_bases = set(range(64))
    return "已选择所有64个DCT基"


def remove_all():
    """取消选择所有基"""
    global selected_bases
    selected_bases = set()
    return "已取消所有选择"


# 自定义CSS样式
custom_css = """
#title {
    text-align: center;
    font-size: 1.8em;
    font-weight: bold;
    margin-bottom: 0.5em;
}
.matrix-display {
    font-family: monospace;
    font-size: 11px;
    white-space: pre;
    line-height: 1.4;
}
"""

# 创建Gradio界面
with gr.Blocks(title="Transform Coding: DCT") as demo:
    gr.Markdown("<div id='title'>Transform Coding: DCT</div>")
    
    # 状态组件存储当前图像
    image_state = gr.State(value=None)
    quality_state = gr.State(value=50)
    
    with gr.Row():
        # 左侧：输入图像
        with gr.Column(scale=1):
            gr.Markdown("### Input Image")
            input_image = gr.Image(label="", type="pil", height=300)
        
        # 中间：DCT基选择
        with gr.Column(scale=1):
            gr.Markdown("### Click to select DCT bases")
            dct_grid_display = gr.Image(label="", interactive=True, height=260)
            
            with gr.Row():
                select_all_btn = gr.Button("Set All", size="sm")
                remove_all_btn = gr.Button("Remove All", size="sm")
            
            gr.Markdown("### JPEG Quality Factor")
            quality_slider = gr.Slider(1, 100, value=50, step=1, label="")
            
            selected_info = gr.Textbox(label="", value="已选择 64/64 个DCT基", interactive=False)
        
        # 右侧：输出图像
        with gr.Column(scale=1):
            gr.Markdown("### Output Image")
            with gr.Row():
                output_image = gr.Image(label="", height=300)
                psnr_display = gr.Textbox(label="", value="PSNR: -- dB", interactive=False)
    
    # 下方：矩阵显示区域
    gr.Markdown("---")
    gr.Markdown("### Selected Block")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_pixels_display = gr.Textbox(
                label="Input Y Pixel Values",
                lines=10,
                interactive=False,
                elem_classes=["matrix-display"]
            )
        
        with gr.Column(scale=1):
            dct_coeffs_display = gr.Textbox(
                label="DCT Coefficients",
                lines=10,
                interactive=False,
                elem_classes=["matrix-display"]
            )
        
        with gr.Column(scale=1):
            quantized_display = gr.Textbox(
                label="Quantised DCT Coefficients",
                lines=10,
                interactive=False,
                elem_classes=["matrix-display"]
            )
    
    with gr.Row():
        with gr.Column(scale=1):
            dequantized_display = gr.Textbox(
                label="Dequantised DCT Coefficients",
                lines=10,
                interactive=False,
                elem_classes=["matrix-display"]
            )
        
        with gr.Column(scale=1):
            inv_transform_display = gr.Textbox(
                label="Inverse Transform Coefficients",
                lines=10,
                interactive=False,
                elem_classes=["matrix-display"]
            )
        
        with gr.Column(scale=1):
            output_pixels_display = gr.Textbox(
                label="Output Y Pixel Values",
                lines=10,
                interactive=False,
                elem_classes=["matrix-display"]
            )
    
    # 处理函数
    def handle_image_change(img):
        global current_image
        current_image = img
        shape_info = getattr(img, "size", None)
        debug_log(f"事件: handle_image_change, image_size={shape_info}")
        return process_image(img, "", current_quality, None)[1:]
    
    def handle_quality_change(q):
        global current_quality
        current_quality = q
        debug_log(f"事件: handle_quality_change, q={q}")
        return process_image(current_image, "", q, None)[1:]
    
    def handle_dct_click(evt: gr.SelectData):
        """处理DCT网格点击"""
        debug_log(f"事件: handle_dct_click, index={evt.index}")
        return process_image(current_image, "", current_quality, (evt.index[0], evt.index[1]))[1:]
    
    def handle_select_all():
        global selected_bases
        selected_bases = set(range(64))
        debug_log("事件: handle_select_all")
        result = process_image(current_image, "", current_quality, None)
        return result[1:]
    
    def handle_remove_all():
        global selected_bases
        selected_bases = set()
        debug_log("事件: handle_remove_all")
        result = process_image(current_image, "", current_quality, None)
        return result[1:]
    
    # 图像上传时更新
    input_image.change(
        fn=handle_image_change,
        inputs=[input_image],
        outputs=[
            dct_grid_display, output_image,
            input_pixels_display, dct_coeffs_display, quantized_display,
            dequantized_display, inv_transform_display, output_pixels_display,
            psnr_display, selected_info
        ]
    )
    
    # 质量滑块改变时更新
    quality_slider.change(
        fn=handle_quality_change,
        inputs=[quality_slider],
        outputs=[
            dct_grid_display, output_image,
            input_pixels_display, dct_coeffs_display, quantized_display,
            dequantized_display, inv_transform_display, output_pixels_display,
            psnr_display, selected_info
        ]
    )
    
    # DCT网格点击事件
    dct_grid_display.select(
        fn=handle_dct_click,
        outputs=[
            dct_grid_display, output_image,
            input_pixels_display, dct_coeffs_display, quantized_display,
            dequantized_display, inv_transform_display, output_pixels_display,
            psnr_display, selected_info
        ]
    )
    
    # 按钮事件
    select_all_btn.click(
        fn=handle_select_all,
        outputs=[
            dct_grid_display, output_image,
            input_pixels_display, dct_coeffs_display, quantized_display,
            dequantized_display, inv_transform_display, output_pixels_display,
            psnr_display, selected_info
        ]
    )
    
    remove_all_btn.click(
        fn=handle_remove_all,
        outputs=[
            dct_grid_display, output_image,
            input_pixels_display, dct_coeffs_display, quantized_display,
            dequantized_display, inv_transform_display, output_pixels_display,
            psnr_display, selected_info
        ]
    )

if __name__ == "__main__":
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    # 不要改我定的端口,因为7860有其他服务在运行
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7857"))
    debug_log(
        f"启动参数: server_name={server_name}, server_port={server_port}, debug={DEBUG_ENABLED}"
    )
    print("🚀 启动 DCT Transform Coding 界面...")
    print(f"📍 访问地址: http://localhost:{server_port}")
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=custom_css,
    )
