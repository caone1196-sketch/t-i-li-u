# 📘 Workflow WAI-illustrious v17 — 3-pass tối ưu ít lỗi nhất

> File: **`workflow_wai_toi_uu.json`** — 27 node
> Model yêu cầu:
> - `WAI-illustrious.safetensors` (checkpoints)
> - `bbox/face_yolov8m.pt` (ultralytics/bbox)
> - `bbox/hand_yolov8s.pt` (ultralytics/bbox)
> - `bbox/foot_anime_yolo11m_v3.pt` (ultralytics/bbox) — *đã fix link trong Cell 2*
> - `sam_vit_b_01ec64.pth` (sams)
> Custom node yêu cầu:
> - **ComfyUI-Impact-Pack + ComfyUI-Impact-Subpack** (cho FaceDetailer, UltralyticsDetectorProvider, SAMLoader)
> - Các node còn lại **đều có sẵn trong ComfyUI gốc** (không cần cài thêm):
>   - ✅ `CheckpointLoaderSimple`, `CLIPTextEncode`, `KSampler`, `LatentUpscale`, `VAEDecodeTiled`, `EmptyLatentImage`, `SaveImage`
>   - ✅ `ConditioningSetTimestepRange`, `ConditioningCombine` (built-in)
>   - ✅ **`FreeU_V2`** (built-in từ ComfyUI 2024-07 — KHÔNG phải FreeUAdvanced của pack custom)
>   - ✅ **`RescaleCFG`** (built-in từ ComfyUI 2024-05 — chống cháy sáng ở CFG cao)
>
> Nếu mở lên vẫn đỏ node `FreeU_V2` hoặc `RescaleCFG` → ComfyUI của bạn quá cũ → vào ComfyUI Manager → **Update ComfyUI**, hoặc xóa 2 node đó rồi nối thẳng `CheckpointLoader → KSampler` (ảnh sẽ hơi cháy sáng một chút nhưng vẫn chạy được).

---

## 🔧 Đã sửa gì so với workflow bạn gửi

| Lỗi ở bản cũ | Đã fix |
|---|---|
| Tên YOLO bị Markdown `face_[yolov8m.pt](http://yolov8m.pt)` | Đổi về đúng `bbox/face_yolov8m.pt`, `bbox/hand_yolov8s.pt`, `bbox/foot_anime_yolo11m_v3.pt` |
| Dây bị gãy: ②→③ thiếu link, model→③ thiếu link | Nối lại đầy đủ mọi dây, đã validation tự động |
| KSampler ③ (siêu nét) nhận prompt thiếu tay/mắt | Đã nối đúng positive = prompt đã gộp (node 17) |
| `force_inpaint = false` ở tay → không chịu sửa | Đặt `force_inpaint = true` cho cả 3 detailer |
| Chân `denoise=0.5, cycle=2` → biến dạng | Giảm còn `0.40, cycle=1` (SAM center-1 cho cắt chính xác hơn) |
| Scheduler ②③ = `karras` với WAI → cháy/nhiễu | Đổi ②③ sang `sgm_uniform` (hợp SDXL hires), ① giữ `karras` cho nháp |
| Mặt nối SAM không cần thiết → đôi khi mask lệch | Mặt bỏ SAM, dùng `feather=20` blend phẳng; tay/chân mới dùng SAM |
| SAM tay `threshold=0.88 dilation=4` → ôm rộng vào nền | Đặt `sam_threshold=0.93, dilation=0, bbox_expansion=4` (chặt hơn) |
| Không có FreeU/Rescale → ảnh hơi mượt quá ở bước upscale | Thêm `FreeU_V2 (b1=1.1, b2=1.2)` + `RescaleCFG (0.75)` → nét hơn, không cháy |
| `VAEDecode` thuần 1248×1824 có thể OSM T4 | Đổi sang `VAEDecodeTiled tile=512` |
| Prompt wildcard điền vào `positive` cho detailer | Đổi: `positive` = chất lượng nền, `wildcard` = từ khóa chuyên biệt (mắt/ngón/ngón chân) → đúng chuẩn ltdrdata |

---

## Luồng 3-pass

```
Checkpoint(WAI) ──► FreeU_V2 ──► RescaleCFG(0.75)
                                          │
  ┌──prompt_chính ──┬──► ConditioningCombine ────┐
  ├──prompt_tây ──►TimestepRange 0.5→1.0───►Combine ─┤
  └──prompt_mắt ──►TimestepRange 0.6→1.0───►Combine ┴── positive ──┐
  ┌──negative───────────────────────────────────────────────────────┤
  └──EmptyLatent 832x1216──────────────────────────────────────────┴──► KSampler ①
                                                                             │
                                                                  LatentUpscale ×1.5 bicubic
                                                                             │ 1248×1824
                                                                             ▼
                                                                      KSampler ② (fix lỗi d=0.28)
                                                                             │
                                                                             ▼
                                                                      KSampler ③ (siêu nét d=0.32)
                                                                             │
                                                                             ▼
                                                                      VAEDecodeTiled
                                                                             │
                                              ┌──────────────────────────────┘
                                              ▼
                                    FaceDetailer ④ MẶT (KO SAM, d=0.28, feather=20)
                                              │
                                              ▼
                                    FaceDetailer ⑤ TAY (SAM_b, rect-4, d=0.30)
                                              │
                                              ▼
                                    FaceDetailer ⑥ CHÂN (SAM_b, center-1, d=0.40, cycle=1)
                                              │
                                              ▼
                                         SaveImage
```

### Thông số từng bước

| Bước | Kích thước | Sampler | Scheduler | Steps | CFG | Denoise |
|---|---|---|---|---|---|---|
| ① Nháp | 832×1216 | euler_ancestral | karras | 28 | 5.0 | 1.00 |
| ② Sửa lỗi latent | 1248×1824 (upscale ×1.5 bicubic) | dpmpp_2m | sgm_uniform | 18 | 5.0 | 0.28 |
| ③ Siêu nét | 1248×1824 | dpmpp_2m | sgm_uniform | 20 | 5.5 | 0.32 |

- `FreeU_V2 (b1=1.1, b2=1.2)`: giảm over-smoothing ở bước upscale, ra ảnh nét hơn.
- `RescaleCFG (0.75)`: CFG thực tế = 5.5 × 0.75 ≈ 4.1 → tránh cháy sáng mà vẫn bám prompt.
- Timestep range cho prompt tay/mắt: tay chỉ ảnh hưởng 50% steps cuối, mắt 40% cuối → không can thiệp vào tổng thể bố cục.

### Thông số 3 detailer

| Bộ phận | SAM | guide | denoise | feather | bbox_thresh | bbox_dilation | crop | sam_hint | sam_thresh | cycle |
|---|---|---|---|---|---|---|---|---|---|---|
| Mắt/mặt | ❌ | 768 | 0.28 | 20 | 0.50 | 8 | 3.0 | — | — | 1 |
| Tay/móng | ✅ vit_b | 768 | 0.30 | 14 | 0.45 | 8 | 2.8 | rect-4 | 0.93 | 1 |
| Chân/móng | ✅ vit_b | 768 | 0.40 | 16 | 0.45 | 10 | 3.2 | center-1 | 0.93 | 1 |

- **`force_inpaint = true`** cho cả 3 node — ép inpaint thay vì blend mềm.
- **`wildcard`** (prompt cộng thêm cho từng vật thể trong vùng detected):
  - Mặt: thêm từ khóa mắt/mi/lông mi
  - Tay: thêm 5 ngón + móng hồng
  - Chân: thêm 5 ngón + móng chân hồng
- **`sam_threshold=0.93`** (cao) cho SAM cắt chặt theo biên tay/chân, không dính nền.
- **`bbox_expansion=4`** ở tay để SAM có context quanh ngón tay nhưng không quá nhiều.
- **`noise_mask_feather=20`** → transition mượt ở rìa.

---

## Tip chỉnh nhanh

| Vấn đề | Chỉnh |
|---|---|
| Tay vẫn 6 ngón | Tăng `denoise` tay lên 0.35; thêm `perfect fingers, correct fingers` vào wildcard tay |
| Mặt bị đổi nhìn khác người | Giảm `denoise` mặt xuống 0.22; tăng `feather` lên 24 |
| Chân biến dạng nhiều | Giảm `denoise` chân xuống 0.35; đổi `sam_detection_hint` sang `rect-4` |
| Ảnh bị cháy sáng/đậm quá | Giảm `RescaleCFG.multiplier` xuống 0.65 |
| Ảnh bị mờ/nhạt | Tăng `multiplier` lên 0.85 hoặc bỏ `RescaleCFG` (xóa node 3, nối 2→KSampler) |
| Tay/chân bị lem vào nền | Giảm `bbox_dilation` tay/chân xuống 4; tăng `sam_threshold` lên 0.95 |
| Không thấy tay/chân bị phát hiện | Giảm `bbox_threshold` tay/chân xuống 0.35 |
| Lỗi "FreeU_V2 not found" / "RescaleCFG not found" | Update ComfyUI (Manager → Update ComfyUI → Restart). Hoặc tạm xóa node 2,3 rồi nối thẳng 1→KSampler.model |
| OOM / đỏ VRAM | Đổi `VAEDecodeTiled.tile_size` xuống 384; hoặc Cell 3 đặt `VRAM=lowvram` |
| Chỉ muốn sửa lỗi trên ảnh có sẵn (không vẽ mới) | Thay `EmptyLatentImage + KSampler ①②③` bằng `LoadImage → VAEEncode`; đặt KSampler đầu tiên `denoise=0.5-0.7` rồi chạy tiếp detailer |

Thời gian chạy dự kiến trên T4:
- ① nháp ~25s
- ② fix lỗi ~35s (latent hires 1248×1824)
- ③ siêu nét ~35s
- VAEDecodeTiled ~8s
- 3 detailer ~30s
- **Tổng ~2 phút/ảnh** (chất lượng cao hơn hẳn bản 1 pass, tỉ lệ ngón lỗi giảm mạnh)

Chúc ảnh ra đẹp! 🎨
