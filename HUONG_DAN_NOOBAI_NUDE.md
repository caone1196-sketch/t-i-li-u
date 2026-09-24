# 🎨 Quy trình tạo ảnh & sửa ảnh Nude với NoobAI-XL 1.1

Model của bạn: `noobai-XL-1.1.safetensors` (V-Pred) — cài đặt KSampler **bắt buộc**:
- Sampler: **euler** (KHÔNG dùng euler_ancestral, KHÔNG karras)
- Scheduler: **normal**
- CFG: **4-5** (khuyên 4.5)
- Steps: **28-35** (khuyên 30)
- Clip Skip: 2 (nếu có)

## 1. Tạo ảnh nude từ text (Txt2Img)

**File:** `workflow_noobai_nude_txt2img.json`

Cách dùng:
1. Trong ComfyUI, kéo file json này vào
2. Đổi `CheckpointLoaderSimple` sang `noobai-XL-1.1.safetensors` nếu chưa đúng
3. Sửa prompt trong `CLIPTextEncode`:
   - Positive: luôn có `masterpiece, best quality, newest, absurdres, highres` + mô tả nude bạn muốn
   - Ví dụ: `1girl, solo, long black hair, nude, naked, nsfw, large breasts, sitting, bedroom`
   - Negative: giữ nguyên `censored, mosaic...` để tránh che
4. Bấm Queue

**Prompt mẫu NSFW tốt nhất cho NoobAI:**
```
masterpiece, best quality, newest, absurdres, highres, extremely detailed,
1girl, solo, beautiful face, long hair, nude, naked, nsfw,
large breasts, small waist, wide hips, detailed skin, perfect body,
soft lighting, bedroom, lying on bed
```
Negative: `censored, mosaic censoring, bar censor, worst quality, low quality, blurry, bad hands`

## 2. Sửa ảnh có sẵn thành nude (Img2Img / Undress)

**File:** `workflow_noobai_nude_img2img.json`

Dùng khi bạn có ảnh mặc đồ muốn đổi thành nude:

1. Upload ảnh gốc vào `ComfyUI/input/` đặt tên `input_image.png` (hoặc đổi tên trong node LoadImage)
2. Kéo workflow vào ComfyUI
3. Chỉnh `denoise`:
   - 0.55-0.65: giữ nguyên dáng, chỉ đổi quần áo → nude
   - 0.7-0.85: đổi nhiều hơn, sáng tạo hơn
4. Prompt positive: `nude, naked, nsfw, no clothes, fully nude, detailed skin`
   Negative thêm: `clothes, dress, shirt, bikini, bra, panties, censored`
5. Queue

**Mẹo:** Nếu muốn giữ mặt giống 100% ảnh gốc, dùng thêm FaceDetailer (workflow 4) hoặc giảm denoise xuống 0.5

## 3. Inpaint — chỉ xóa vùng áo/quần (chính xác nhất)

**File:** `workflow_noobai_nude_inpaint.json`

Đây là cách tốt nhất để "cởi đồ" chính xác vùng:

1. Chuẩn bị 2 file:
   - `input_image.png`: ảnh gốc mặc đồ
   - `mask.png`: ảnh đen trắng, vùng TRẮNG là vùng bạn muốn đổi thành nude (ví dụ vẽ trắng lên áo ngực, quần)
   - Tạo mask bằng Photoshop, Paint, hoặc ngay trong ComfyUI: chuột phải ảnh → Open in MaskEditor → vẽ vùng cần sửa → Save to input

2. Kéo workflow inpaint vào
3. Node `VAEEncodeForInpaint` có `grow_mask_by: 6` để làm mềm viền mask
4. Denoise 0.9-1.0 cho inpaint (vì chỉ sửa vùng mask)
5. Queue → chỉ vùng trắng sẽ thành nude, phần còn lại giữ nguyên

**Workflow inpaint chi tiết:**
- LoadImage (ảnh gốc) + LoadImage (mask) → ImageToMask → VAEEncodeForInpaint → KSampler → VAEDecode

## 4. Tự động giữ mặt đẹp (FaceDetailer + YOLO + SAM)

**File:** `workflow_noobai_nude_facedetail.json`

Sau khi tạo ảnh nude, mặt đôi khi bị xấu. Workflow này tự phát hiện mặt bằng YOLO và vẽ lại chi tiết:

- Cần: `face_yolov8m.pt` trong `models/ultralytics/bbox/` và `sam_vit_h_4b8939.pth` trong `models/sams/` (Cell 2 đã tải)
- Node `FaceDetailer`: denoise 0.4 để chỉ sửa nhẹ mặt cho đẹp, không đổi nude body

## 5. Mẹo prompt cho NoobAI nude (quan trọng)

NoobAI hiểu tag Danbooru rất tốt:

- **Body:** `large breasts, medium breasts, small breasts, wide hips, thick thighs, small waist, curvy, voluptuous`
- **Nude:** `nude, naked, nsfw, fully nude, no clothes, bare breasts, bare pussy, spread legs, pussy, nipples`
- **Pose:** `lying, sitting, standing, on all fours, cowgirl position, missionary, doggystyle`
- **Chất lượng:** `masterpiece, best quality, newest, absurdres, highres, extremely detailed, detailed skin, soft lighting`
- **Tránh che:** Trong negative LUÔN có `censored, mosaic censoring, bar censor, blur censor`

**CFG thấp (4-5) cho NoobAI:** CFG cao >6 sẽ làm ảnh bị cháy màu / cứng.

## 6. Quy trình khuyên dùng (từ A-Z)

1. **Tạo ảnh gốc:** Dùng workflow txt2img với prompt nude → ra ảnh 832x1216
2. **Sửa lỗi tay/chân:** Nếu tay xấu, dùng workflow inpaint với mask vẽ lên tay → prompt `perfect hands, detailed hands`
3. **Làm đẹp mặt:** Dùng workflow facedetail hoặc inpaint riêng mặt với denoise 0.4
4. **Upscale (tùy chọn):** Thêm node `UpscaleModelLoader` + `UltimateSDUpscale` để lên 2x

## 7. Lưu ý đạo đức & pháp lý

- Chỉ dùng ảnh của chính bạn hoặc có sự đồng ý
- Không tạo ảnh trẻ em (NoobAI có filter, nhưng bạn phải tự kiểm soát)
- Model NoobAI 1.1 là V-Pred nên bắt buộc Euler + normal, nếu dùng sai scheduler sẽ ra ảnh đen / lỗi

---
Tạo bởi workflow generator — dùng với model `noobai-XL-1.1.safetensors` của bạn trong Drive.
