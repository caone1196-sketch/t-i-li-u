# 📸 FLUX Q5_K_S REALISTIC V2 BEST - Tối ưu nhất, fix mọi lỗi

> **File khuyên dùng:** `workflow_flux_schnell_Q5_realistic_v2_best.json` - 15 node, chạy 100% trên cả 2 notebook Free và WAI_fixed

## 🔥 V2 BEST khác gì V1?

| Lỗi V1 gặp | V2 BEST đã fix |
|---|---|
| Dùng `hand_yolov8s.pt` - Free notebook không có | Đổi sang `hand_yolov8m.pt` - có ở cả 2 notebook |
| FaceDetailer tay nối SAM -> OOM trên T4 lowvram | **Bỏ SAM ở tay**, dùng BBOX + feather 20 (ổn định hơn, không lem) |
| `sam_vit_b` không có ở Free notebook (chỉ có `sam_vit_h`) | Cell 1B và Cell 2 đã update để tải cả 2 SAM, workflow ưu tiên `sam_vit_b` |
| `denoise` mặt 0.15 quá cao -> đổi mặt | Giảm xuống **0.12** - giữ identity tốt nhất |
| Không có bản không SAM dự phòng | Thêm `workflow_flux_schnell_Q5_realistic_no_sam.json` |
| Không có bản minimal test | Thêm `workflow_flux_schnell_Q5_realistic_minimal.json` chỉ 8 node |

## 📦 Workflow khuyên dùng theo thứ tự test

### 1. Test nhanh (8 node) - `minimal`
```
UnetLoaderGGUF + DualCLIPLoaderGGUF + VAELoader + EmptyLatent 832x1216 + KSampler 4 steps + VAEDecode + Save
```
- Dùng để test xem GGUF có load được không
- Nếu chạy được -> model OK, chuyển sang V2 BEST
- Nếu đỏ node `UnetLoaderGGUF` -> chưa cài ComfyUI-GGUF -> chạy lại Cell 1B mới

### 2. V2 BEST (15 node) - `v2_best` - KHUYÊN DÙNG
```
... + FaceDetector face_yolov8m + HandDetector hand_yolov8m + SAMLoader sam_vit_b
+ FaceDetailer MẶT (có SAM, denoise 0.12, guide 1024, feather 5)
+ FaceDetailer TAY (KHÔNG SAM, denoise 0.22, guide 512, feather 20, drop_size 30)
```
- Mặt: có SAM cắt chuẩn, denoise thấp giữ mặt gốc
- Tay: KHÔNG SAM, BBOX + feather 20 blend mượt, không lem vào đùi/nền (bí kíp từ bản anime)
- Tốc độ: ~25-30s/ảnh trên T4 lowvram

### 3. No SAM (15 node) - `no_sam`
- Giống V2 BEST nhưng cả mặt và tay đều KHÔNG SAM
- Dùng khi SAM tải lỗi hoặc OOM

### 4. Hires (17 node) - `hires`
- V2 BEST + LatentUpscale x1.5 + KSampler denoise 0.35
- Ra ảnh 1248x1824 ~2K, nét hơn nhưng chậm hơn 40%

## ⚙️ Thông số vàng V2 BEST

### KSampler gốc
```
steps: 4 (Schnell chuẩn, đừng tăng)
cfg: 1.0 (bắt buộc)
sampler: euler
scheduler: simple
size: 832x1216 (dọc 2:3 đẹp nhất cho người)
```

### FaceDetailer MẶT (có SAM)
```
detector: bbox/face_yolov8m.pt
sam: sam_vit_b_01ec64.pth (hoặc sam_vit_h_4b8939.pth nếu không có b)
denoise: 0.12 (giữ identity, chỉ làm nét da/mắt)
guide_size: 1024 (cần lớn để thấy lỗ chân lông)
bbox_threshold: 0.5
bbox_dilation: 10
crop_factor: 3.0
feather: 5 (vì có SAM cắt chuẩn nên feather thấp)
sam_threshold: 0.93
wildcard: detailed skin texture, natural pores, sharp eyes, detailed iris, photorealistic skin
```

### FaceDetailer TAY (KHÔNG SAM - bí kíp)
```
detector: bbox/hand_yolov8m.pt (hoặc hand_yolov8s.pt)
sam: KHÔNG NỐI (để trống)
denoise: 0.22
guide_size: 512
bbox_threshold: 0.35 (hạ thấp để bắt tay khuất)
bbox_dilation: 12 (rộng hơn để ôm trọn bàn tay)
crop_factor: 2.8
feather: 20 (QUAN TRỌNG: blend mềm viền không có SAM)
drop_size: 30 (bỏ phát hiện nhỏ nhầm)
wildcard: five fingers, perfect hands, detailed fingernails, realistic skin texture
```

**Tại sao tay KHÔNG SAM lại tốt hơn?**
- SAM train trên người thật nhưng khi tay đặt trên đùi/áo, SAM hay ôm cả mảng đùi vào mask -> detailer vẽ hỏng
- BBOX phẳng + feather 20: ở giữa mask 100% inpaint, ở viền giảm dần về 0 -> không lem

## 📝 Prompt người thật chuẩn Flux

Flux thích câu tự nhiên, không thích tag soup.

❌ Xấu:
```
1girl, solo, absurdres, aesthetic, long hair
```

✅ Tốt:
```
A beautiful young Vietnamese woman, 24 years old, natural skin texture with visible pores, soft natural makeup, long straight black hair, wearing an elegant white summer dress, standing in a blooming garden, soft morning sunlight, photorealistic, ultra detailed face, detailed skin, sharp focus, DSLR photo, 85mm lens, f1.8, bokeh background, 8k, detailed hands, five fingers, hands visible, full body, natural pose, realistic skin texture
```

**Từ khóa bắt buộc cho da thật:**
- `natural skin texture, visible pores, realistic skin texture`
- `DSLR photo, 85mm lens, bokeh, soft natural lighting`
- `detailed hands, five fingers, hands visible` (nếu thấy tay)

**Negative:**
```
cartoon, anime, illustration, painting, 3d render, blurry, low quality, deformed, extra fingers, mutated hands, bad anatomy, watermark, text, oversaturated, plastic skin, airbrushed, doll
```

## ❌ Fix lỗi thường gặp

| Lỗi | Nguyên nhân | Fix |
|---|---|---|
| **Đỏ node UnetLoaderGGUF / DualCLIPLoaderGGUF** | Chưa cài ComfyUI-GGUF | Chạy lại Cell 1B mới (đã thêm clone ComfyUI-GGUF) |
| **Đỏ node FaceDetailer / UltralyticsDetectorProvider** | Chưa cài Impact Pack | Chạy lại Cell 1B |
| **Model not found: hand_yolov8s.pt** | Free notebook cũ không tải s | Dùng V2 BEST dùng `hand_yolov8m.pt` hoặc chạy lại Cell 2 mới (đã thêm hand_yolov8s) |
| **Model not found: sam_vit_b_01ec64.pth** | Free notebook cũ chỉ có sam_vit_h | Chạy lại Cell 2 mới (đã thêm sam_vit_b) hoặc đổi SAMLoader sang `sam_vit_h_4b8939.pth` |
| **CUDA OOM** | VRAM đầy | Cell 3 đặt `VRAM=lowvram`, `VAE_PREC=cpu-vae`, `PREVIEW=none`, dùng bản minimal trước |
| **Ảnh đen / nhiễu** | Model chưa load xong | Đợi 30s rồi Queue lại |
| **Mặt bị đổi khác người** | denoise cao quá | Giảm denoise mặt xuống 0.10-0.12, tăng feather lên 12 |
| **Tay 6 ngón vẫn lỗi** | denoise thấp quá hoặc seed xấu | Tăng denoise tay lên 0.28, thử seed khác, wildcard thêm `correct anatomy` |
| **Da nhựa như búp bê** | Thiếu từ khóa da | Thêm `natural skin texture, pores` vào prompt + wildcard |

## 🚀 Quy trình chạy đúng (Colab)

1. **Cell 1:** Mount Drive, cài ComfyUI
2. **Cell 1B mới:** Cài Impact Pack + Subpack + **ComfyUI-GGUF** + symlink vae/clip/unet (đã fix)
3. **Cell 2 mới:** Tải FLUX Q5_K_S + T5 Q4 + VAE + CLIP + YOLO face/hand + SAM b + SAM h (đã thêm hand_yolov8s và sam_vit_b)
4. **Cell 3:** Chạy ComfyUI `lowvram`, `fp16-vae`, `pytorch` attention
5. **Cell 3C mới:** Tải workflow V2 BEST (tự tìm branch mới)
6. **Test:** Load `workflow_flux_schnell_Q5_realistic_minimal.json` trước -> Queue -> nếu OK thì load `workflow_flux_schnell_Q5_realistic_v2_best.json`

## 📂 File đính kèm (tối ưu nhất)

- `workflow_flux_schnell_Q5_realistic_minimal.json` - 8 node, test nhanh
- `workflow_flux_schnell_Q5_realistic_v2_best.json` - 15 node, **KHUYÊN DÙNG**
- `workflow_flux_schnell_Q5_realistic_no_sam.json` - 15 node, không cần SAM
- `workflow_flux_schnell_Q5_realistic_hires.json` - 17 node, upscale 2K
- `workflow_flux_schnell_Q5_realistic_inpaint.json` - inpaint tay/mặt lỗi
- `QUY_TRINH_FLUX_REALISTIC_V2_BEST.md` - file này

Chúc ảnh người thật đẹp như DSLR! 📸
