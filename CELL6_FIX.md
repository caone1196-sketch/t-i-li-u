# 🔧 FIX CELL 6 - Lỗi Gradio Inpaint FLUX GGUF

## ❌ Lỗi cũ gặp phải

Cell 6 gốc (trong `ComfyUI_Colab_WAI_fixed.ipynb`) bị lỗi vì:

1. **Gradio API cũ:** Dùng `gr.Image(tool="sketch")` - param `tool` đã bị xóa từ Gradio 4.0. Gradio 4/5 yêu cầu `gr.ImageEditor` hoặc `gr.Sketchpad`
2. **Branch cũ:** `BRANCH = "arena/01a0cc76-t-i-li-u"` không còn tồn tại, tải workflow 404
3. **Mask handling sai:** `ImageToMask channel=red` nhưng mask lưu dạng L (grayscale) -> ra mask đen hoàn toàn
4. **Default image:** `gr.Image(value="/path/to/file")` với `type="pil"` - Gradio 4 không nhận path string, phải load PIL
5. **Launch param lỗi:** `demo.launch(inline=False)` - `inline` đã bỏ từ Gradio 4
6. **Không check ComfyUI:** Nếu Cell 3 chưa chạy, cell 6 vẫn chạy và báo lỗi khó hiểu

## ✅ Đã fix gì (bản 2026-09-24)

### 1. Tương thích Gradio 3,4,5
```python
has_image_editor = hasattr(gr, 'ImageEditor')
has_sketchpad = hasattr(gr, 'Sketchpad')
has_image_mask = hasattr(gr, 'ImageMask')

if has_image_editor:  # Gradio 5
    img_in = gr.ImageEditor(..., brush=gr.Brush(...))
elif has_sketchpad:   # Gradio 4
    img_in = gr.Sketchpad(...)
elif has_image_mask:
    img_in = gr.ImageMask(...)
else:  # Gradio 3 fallback
    img_in = gr.Image(tool="sketch", ...)
```

### 2. Parse mask đa dạng
Hàm `parse_gradio_mask()` xử lý tất cả kiểu return:
- Dict `{'image': PIL, 'mask': PIL}` (Gradio 3)
- Dict `{'background': ..., 'layers': [...], 'composite': ...}` (Gradio 5 ImageEditor)
- Tuple `(image, mask)` (Gradio 4 ImageMask)
- Kiểm tra mask rỗng, mask toàn trắng, resize mask bằng background

### 3. Branch tự động
```python
BRANCHES_TO_TRY = ["arena/01a0d36a-t-i-li-u", "main", ...]
for br in BRANCHES_TO_TRY:
    r = requests.head(f".../{br}/workflow_flux_schnell_Q5_realistic.json")
    if r.status_code == 200:
        BRANCH = br; break
```

### 4. Workflow tối ưu người thật
- Prompt mặc định: `detailed hand, five fingers, natural skin texture, photorealistic`
- Negative: `cartoon, anime, blurry, deformed, plastic skin`
- Denoise 0.5 (giữ bố cục), grow_mask 12px, steps 8, cfg 1.0
- Dùng `UnetLoaderGGUF + DualCLIPLoaderGGUF + VAELoader` đúng chuẩn FLUX

### 5. Check ComfyUI + log rõ ràng
```python
def check_comfy():
    r = requests.get(f"{COMFY}/system_stats", timeout=5)
    return r.status_code == 200
```

### 6. Fix launch
```python
demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7860)
# bỏ inline=False
```

## 🚀 Cách dùng sau fix

1. Chạy **Cell 1 → 1B → 2 → 3** (đợi link cloudflared hiện)
2. Chạy **Cell 6** mới:
   - Sẽ tự hiện `Gradio version: 5.x, has ImageEditor=True`
   - Nếu thấy `❌ ComfyUI chưa chạy` → chạy lại Cell 3
3. Upload ảnh + **tô trắng** vùng lỗi (bản mới tô trắng, bản cũ tô đen)
4. Prompt ví dụ:
   - Mặt: `photorealistic face, natural skin texture, visible pores, sharp eyes`
   - Tay: `five fingers, perfect hands, detailed fingernails, realistic skin`
5. Bấm **Sửa vùng tô** → chờ 15-25s

## 📂 File liên quan

- `ComfyUI_Colab_WAI_fixed.ipynb` - Cell 9 (CELL 6) đã fix
- `ComfyUI_Colab_Free.ipynb` - Cell 5 (CELL 3B) và Cell 6 (CELL 3C) đã fix branch
- `workflow_flux_schnell_Q5_realistic_inpaint.json` - workflow inpaint riêng cho người thật
- `giaodien_flux_realistic.py` - UI tạo ảnh người thật (dùng cho Cell 3B)

## 💡 Nếu vẫn lỗi

| Lỗi | Fix |
|---|---|
| `ModuleNotFoundError: gradio` | Cell tự cài `pip install gradio`, nếu vẫn lỗi chạy `!pip install -q gradio` tay |
| `Connection refused 127.0.0.1:8188` | Cell 3 chưa chạy xong, đợi cổng 8188 mở |
| `Mask rỗng` | Bạn chưa tô gì, dùng brush trắng tô lên tay/mặt lỗi |
| `404 workflow` | Branch cũ, đã fix tự tìm branch mới, chạy lại Cell 3C |
| `CUDA OOM` | Cell 3 đặt `VRAM=lowvram`, `VAE_PREC=cpu-vae` |

