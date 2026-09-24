# 📸 QUY TRÌNH FLUX.1-schnell Q5_K_S — TỐI ƯU CHO NGƯỜI THẬT (Photorealistic)

> **File workflow chính:** `workflow_flux_schnell_Q5_realistic.json` — 18 node, chạy mượt trên T4 16GB Colab Free
> **File upscale:** `workflow_flux_schnell_Q5_realistic_hires.json` — thêm bước Latent Upscale x1.5 để ra ảnh 2K
> **Model gốc:** `flux1-schnell-Q5_K_S.gguf` (8.26GB Q5_K_S — chất lượng cao nhất dưới 15GB)

Workflow này được viết lại hoàn toàn từ bản anime để tối ưu cho **da thật, mặt thật, tay chân thật**.

---

## 🔥 Khác biệt so với bản anime cũ

| Vấn đề bản anime | Bản realistic đã fix |
|---|---|
| Prompt tag kiểu Danbooru `1girl, absurdres, aesthetic` | Dùng **câu tự nhiên tiếng Anh** như mô tả cho photographer: `A beautiful Vietnamese woman, natural skin texture...` |
| `foot_anime_yolo11m_v3.pt` chỉ nhận chân anime | Đổi sang `person_yolov8m.pt` hoặc bỏ, chỉ giữ `face` + `hand` — SAM trên người thật cắt cực chuẩn |
| FaceDetailer **không nối SAM** ở tay/chân | **BẬT SAM cho cả tay** vì SAM được train trên người thật → mask tay siêu chính xác, không lem |
| `denoise` mặt 0.18 quá cao → đổi mặt người | Giảm `denoise` mặt xuống **0.12-0.18**, `feather=8-12` để giữ nguyên identity |
| `guide_size=512` quá nhỏ → mắt mờ | Tăng `guide_size` mặt lên **1024**, tay 512 → đủ chi tiết lỗ chân lông |
| Kích thước vuông 1024x1024 | Đổi mặc định **832x1216 (dọc)** hoặc **1024x1536** — tỉ lệ người thật đẹp nhất |
| Không có negative | Thêm negative nhẹ: `cartoon, anime, illustration, blurry, extra fingers...` (Flux schnell vẫn hiểu) |

---

## 📦 Model yêu cầu (Tổng ~12.3GB)

| File | Size | Đặt ở đâu trong ComfyUI |
|---|---|---|
| `flux1-schnell-Q5_K_S.gguf` | 8.26 GB | `models/unet/` |
| `t5-v1_1-xxl-encoder-Q4_K_M.gguf` | 2.9 GB | `models/clip/` |
| `clip_l.safetensors` | 250 MB | `models/clip/` |
| `ae.safetensors` (FLUX VAE) | 320 MB | `models/vae/` |
| `face_yolov8m.pt` | 52 MB | `models/ultralytics/bbox/` |
| `hand_yolov8s.pt` hoặc `hand_yolov8n.pt` | 22 MB | `models/ultralytics/bbox/` |
| `sam_vit_b_01ec64.pth` | 375 MB | `models/sams/` |
| **TỔNG** | **~12.3GB** | |

> **Custom nodes bắt buộc:**
> - `ComfyUI-GGUF` (để có `UnetLoaderGGUF`, `DualCLIPLoaderGGUF`)
> - `ComfyUI-Impact-Pack` + `ComfyUI-Impact-Subpack` (FaceDetailer, Detector)
> - `ComfyUI-Manager` (để update)

---

## ⚙️ Thông số vàng cho người thật

### 1. KSampler gốc (Tạo ảnh)

```
sampler_name: euler
scheduler: simple
steps: 4         # Schnell chuẩn là 4, đừng tăng!
cfg: 1.0         # Schnell BẮT BUỘC cfg=1
denoise: 1.0
width: 832
height: 1216     # Dọc 2:3 là đẹp nhất cho chân dung / full body
seed: randomize
```

> **Mẹo:** Muốn ảnh nét hơn, đừng tăng steps lên 8-20 (sẽ bị over-smooth). Hãy giữ 4 steps và dùng FaceDetailer + Upscale.

### 2. Prompt cho người thật (QUAN TRỌNG NHẤT)

**Flux KHÔNG thích tag soup như SDXL. Flux thích câu văn tự nhiên.**

#### ❌ Prompt xấu (kiểu anime):
```
1girl, solo, absurdres, very aesthetic, long hair, school uniform
```

#### ✅ Prompt tốt (photorealistic):
```
A beautiful young Vietnamese woman, 24 years old, natural skin texture with visible pores, soft natural makeup, long straight black hair, wearing an elegant white summer dress, standing in a blooming garden, soft morning sunlight, photorealistic, ultra detailed face, sharp focus, DSLR photo, 85mm lens, bokeh background, 8k
```

#### Prompt mẫu theo nhu cầu:

**Chân dung cận mặt:**
```
close-up portrait of a beautiful woman, 26 years old, natural skin, freckles, detailed eyes, soft natural lighting, photorealistic, ultra detailed, pores, subtle makeup, blurred background, 85mm lens
```

**Full body ngoài trời:**
```
full body photo of a young woman, athletic body, wearing casual jeans and white t-shirt, standing in a modern city street, natural pose, arms at sides, hands visible, natural skin texture, photorealistic, sharp focus, DSLR, detailed hands, five fingers
```

**Beauty / Fashion:**
```
fashion model, 25 years old, flawless natural skin, professional makeup, studio softbox lighting, wearing black evening dress, elegant pose, photorealistic, ultra high resolution, detailed skin texture, sharp eyes
```

#### Negative prompt (để trống cũng được, nhưng nên có):
```
cartoon, anime, illustration, painting, 3d render, blurry, low quality, deformed, extra fingers, mutated hands, bad anatomy, watermark, text, oversaturated, plastic skin, airbrushed
```

### 3. FaceDetailer — Mặt người thật

| Tham số | Giá trị khuyên dùng | Lý do |
|---|---|---|
| `bbox_detector` | `face_yolov8m.pt` | Nhận diện mặt thật chuẩn |
| `sam_model_opt` | **CÓ NỐI** `sam_vit_b` | SAM trên người thật cắt cực nét, ôm khít mặt |
| `denoise` | **0.12 - 0.18** | Thấp để giữ identity, chỉ làm nét da/mắt |
| `guide_size` | **1024** | Cần lớn để thấy lỗ chân lông |
| `guide_size_for` | true (Bbox) | |
| `bbox_threshold` | 0.5 | |
| `bbox_dilation` | 10 | Mở rộng 10px để lấy cả tóc mai |
| `bbox_crop_factor` | 3.0 | |
| `sam_detection_hint` | `center-1` | Lấy mặt trung tâm |
| `sam_threshold` | 0.93 | Cắt chặt |
| `feather` | 5-12 | Blend mượt, không để lộ viền |
| `steps` | 6-8 | |
| `cfg` | 1.0 | |
| `wildcard` | `detailed skin texture, natural pores, sharp eyes, detailed eyelashes, natural eyebrows` | |

> **Bí kíp:** Nếu mặt bị đổi thành người khác → giảm denoise xuống 0.10-0.12. Nếu mắt vẫn mờ → tăng lên 0.20.

### 4. FaceDetailer — Tay người thật

| Tham số | Giá trị | Lý do |
|---|---|---|
| `bbox_detector` | `hand_yolov8s.pt` hoặc `hand_yolov8n.pt` | s/n nhạy hơn m với tay nhỏ |
| `sam_model_opt` | **CÓ NỐI SAM** | **Khác bản anime:** người thật SAM cắt tay rất chuẩn |
| `denoise` | **0.20 - 0.25** | Cao hơn mặt, vì tay hay lỗi ngón |
| `guide_size` | 512 | |
| `bbox_threshold` | 0.35 | Hạ xuống 0.35 để bắt được tay khuất |
| `bbox_dilation` | 10 | |
| `bbox_crop_factor` | 2.5 | |
| `wildcard` | `five fingers, perfect hands, detailed fingernails, natural knuckles, realistic skin texture` | |
| `feather` | 5 | |

### 5. Bỏ Detailer chân anime

Với người thật, **không dùng** `foot_anime_yolo11m_v3.pt`. Nếu cần sửa chân, dùng `person_yolov8m.pt` hoặc `hand` cũng bắt được chân, hoặc đơn giản bỏ qua bước chân vì Flux Schnell Q5 đã vẽ chân khá tốt.

---

## 🔄 Workflow đầy đủ (node graph)

```
[1] UnetLoaderGGUF: flux1-schnell-Q5_K_S.gguf
         ↓ MODEL
[2] DualCLIPLoaderGGUF: clip_l + t5-v1_1-xxl-Q4_K_M (type=flux)
         ↓ CLIP
    ┌────┴────┐
[10] Prompt thật    [11] Negative
    ↓ COND         ↓ COND
[20] EmptyLatent 832x1216
         ↓ LATENT + MODEL + COND
[21] KSampler (euler/simple/4 steps/cfg 1.0)
         ↓ LATENT
[22] VAEDecode (vae: ae.safetensors)
         ↓ IMAGE
[30] UltralyticsDetectorProvider face_yolov8m
[31] UltralyticsDetectorProvider hand_yolov8s
[33] SAMLoader sam_vit_b
         ↓
[34] CLIPTextEncode: detailer positive (photorealistic skin)
[35] CLIPTextEncode: detailer negative
         ↓
[40] FaceDetailer FACE (có SAM, denoise 0.15, guide 1024)
         ↓ IMAGE
[41] FaceDetailer HAND (có SAM, denoise 0.22, guide 512)
         ↓ IMAGE
[50] SaveImage -> FLUX_Realistic_Q5_*.png
```

---

## 📐 Kích thước khuyên dùng cho người thật

| Nhu cầu | Width x Height | Ghi chú |
|---|---|---|
| **Chân dung dọc (khuyên dùng)** | 832 x 1216 | Tỉ lệ 2:3 như ảnh thẻ, đẹp nhất |
| Full body cao | 768 x 1344 | Cao hơn, thấy cả chân |
| Bán thân | 1024 x 1024 | Vuông, tập trung mặt/ngực |
| Ngang cinematic | 1216 x 832 | Phong cảnh + người |
| Siêu nét 2K | 1024 x 1536 → upscale x1.5 = 1536 x 2304 | Dùng workflow hires |

> Luôn giữ **bội số của 32**, tổng pixel ~1M-1.5M là ngọt nhất cho Flux.

---

## 💡 10 Mẹo ra ảnh người thật như DSLR

1. **Thêm từ khóa máy ảnh:** `DSLR photo, 85mm lens, f/1.8, bokeh, natural lighting` → da sẽ thật hơn ngay
2. **Nhấn mạnh skin texture:** `natural skin texture, visible pores, subtle skin imperfections` → tránh da nhựa
3. **Ánh sáng:** `soft morning sunlight, window light, studio softbox` → Flux rất nhạy với ánh sáng
4. **Đừng dùng từ anime:** Xóa hết `absurdres, aesthetic, masterpiece` kiểu anime, thay bằng `photorealistic, ultra detailed, 8k, sharp focus`
5. **Tay:** Luôn thêm `hands visible, five fingers, detailed hands` vào prompt chính
6. **Seed:** Nếu tay lỗi, đừng chỉnh denoise vội — thử **random seed khác** trước, Flux Schnell random rất mạnh
7. **Denoise mặt thấp:** Mặt người thật chỉ cần 0.12-0.15, cao quá sẽ đổi mặt
8. **Tắt color grading mạnh:** Tránh `vibrant, highly saturated` → da sẽ giả
9. **Dùng SAM cho người thật:** Đừng sợ SAM lem như anime, trên người thật SAM là vua
10. **Hires:** Nếu muốn in khổ lớn, dùng workflow hires: generate 832x1216 → upscale latent x1.5 → KSampler denoise 0.35 với 6 steps

---

## ⏱ Tốc độ trên T4 16GB lowvram

| Bước | Thời gian |
|---|---|
| Load model lần đầu | 30-45s |
| Generate 4 steps 832x1216 | ~12-18s |
| FaceDetailer mặt (SAM, guide 1024) | ~8-12s |
| FaceDetailer tay (SAM, guide 512) | ~5-7s |
| **Tổng / ảnh** | **~25-35s** |

---

## ❌ Xử lý lỗi

| Lỗi | Cách fix |
|---|---|
| **Da nhựa / quá mịn như búp bê** | Thêm `natural skin texture, pores` vào prompt + wildcard detailer. Giảm denoise mặt xuống 0.12 |
| **Mặt bị đổi khác người** | Giảm denoise mặt xuống 0.10, tăng feather lên 12 |
| **Mắt mờ / lé** | Tăng denoise mặt lên 0.20, guide_size lên 1024, wildcard thêm `sharp eyes, detailed iris` |
| **Tay 6 ngón** | Tăng denoise tay lên 0.28, bbox_threshold hạ xuống 0.30, wildcard thêm `correct anatomy` |
| **Ảnh tối / ám vàng** | Thêm `bright lighting, natural daylight` vào prompt, xóa `warm lighting` |
| **OOM CUDA** | `VRAM=lowvram`, `VAE_PREC=cpu-vae`, `PREVIEW=none`, tắt tab Gradio |
| **SAM không cắt** | Kiểm tra `models/sams/sam_vit_b_01ec64.pth` đã tải chưa, symlink đúng chưa |
| **Không thấy GGUF loader** | Cài `ComfyUI-GGUF` từ Manager, restart ComfyUI |

---

## 📂 File đính kèm

- `workflow_flux_schnell_Q5_realistic.json` — Workflow chính (18 node) — DÙNG FILE NÀY
- `workflow_flux_schnell_Q5_realistic_hires.json` — Bản upscale lên 2K
- `workflow_flux_schnell_gguf_toi_uu.json` — Bản cũ (anime) để so sánh

Chúc bạn ra ảnh người thật đẹp như chụp studio! 📸
