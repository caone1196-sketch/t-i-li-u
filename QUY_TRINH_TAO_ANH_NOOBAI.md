# 📘 Quy trình tạo ảnh NoobAI-XL V-Pred 1.1 (tiếng Việt) — FIX LỖI CẮT SAI VÙNG TAY/CHÂN

> File đi kèm: **`workflow_noobai_toan_dien.json`** (19 node)

---

## 🔧 VẤN ĐỀ: SAM cắt xuyên qua nền → detailer vẽ lem ra ngoài tay/chân

**Nguyên nhân gốc:** `sam_vit_h` được Meta huấn luyện trên **ảnh người thật**, không phải anime. Trên anime:
- Tay nhân vật thường có đường viền không rõ ràng (cùng tông màu da với cánh tay, với đùi khi tay đặt trên đùi)
- SAM dính theo màu/độ tương phản, nó thường **ôm cả một mảng nền/da lớn** vào mask tay/chân.
- Khi detailer inpaint trên một mask quá to, nó không biết "đâu là tay", vẽ ra một cục mỡ/biến dạng ngay vị trí nhầm lẫn.

**Giải pháp lần này:**

| Bộ phận | Có dùng SAM? | Kỹ thuật |
|---|---|---|
| ✨ Mặt | **CÓ** | Mặt anime có đường nét rõ, SAM hoạt động tốt |
| 🤚 Tay | **KHÔNG** | Dùng BBOX phẳng (hình chữ nhật) với **feather=20** — mask mềm ra ở viền để blend tự nhiên, không cần cắt chính xác từng pixel ngón tay |
| 🦶 Chân | **KHÔNG** | Tương tự tay: BBOX phẳng + feather 20 |

### Tại sao BBOX phẳng lại đỡ lỗi hơn SAM?

- Khi dùng SAM, mask bám theo "đường biên" mà SAM tìm được — nếu tìm sai (rất thường với anime tay/chân), mask to ra gấp rưỡi → detailer vẽ hỏng to.
- Khi dùng BBOX phẳng + feather lớn, mask là hình chữ nhật bao trọn tay/chân, viền được làm mờ 20px →:
  - Ở chính giữa vùng tay/chân: mask = 100% → inpaint toàn bộ để sửa ngón
  - Ở viền (rìa BBOX): mask giảm dần về 0 → trộn với ảnh gốc, KHÔNG vẽ lem ra ngoài
  - Quan trọng nhất: **không bao giờ mask ôm nhầm vào đùi/nền** vì BBOX đã được YOLO khoanh chính xác ở nơi có đối tượng.

### Tham số tay/chân mới (KHÔNG SAM):

| Tham số | Mặt (có SAM) | Tay (KHÔNG SAM) | Chân (KHÔNG SAM) |
|---|---|---|---|
| YOLO model | `face_yolov8m.pt` | `hand_yolov8s.pt` (nhẹ, nhạy hơn) | `foot_anime_yolo11m_v3.pt` |
| `sam_model_opt` | ✅ nối SAM | ❌ bỏ trống | ❌ bỏ trống |
| `bbox_threshold` | 0.55 | **0.55** (cao hơn để không bấm nhầm) | 0.50 |
| `bbox_dilation` | 4 | **12** (rộng hơn chút để ôm trọn bàn tay) | 12 |
| `bbox_crop_factor` | 3.0 | 3.0 | 3.0 |
| `denoise` | 0.32 | **0.32** (thấp hơn, chỉ chỉnh nhẹ) | 0.35 |
| `feather` | 8 | **20** (quan trọng nhất: blend mềm viền không có SAM) | 20 |
| `drop_size` | 20 | **30** (bỏ các phát hiện nhỏ/nhầm) | 30 |
| `cycle` | 1 | 1 | 1 |
| prompt | detailed face… | **5 fingers, knuckles, natural fingernails** | **5 toes, natural toenails, ankle** |

### Luồng nối dây

```
   VAEDecode ─► 20✨ FaceDetailer (MẶT) ──► 30🤚 FaceDetailer (TAY, ko SAM) ──► 40🦶 FaceDetailer (CHÂN, ko SAM) ──► Save
                  ▲ bbox: YOLO mặt                   ▲ bbox: YOLO tay                    ▲ bbox: YOLO chân
                  ▲ sam_model_opt: SAM                (KHÔNG nối sam_model_opt)           (KHÔNG nối sam_model_opt)
```

Chú ý: 3 node `30`, `40` KHÔNG có dây `sam_model_opt` nối vào — đó là điểm mấu chốt. Khi không nối `sam_model_opt`, FaceDetailer sẽ dùng BBOX thuần (mờ bằng `feather`) chứ không cắt bằng SAM.

---

## Nếu kết quả vẫn chưa vừa ý — chỉnh từng bước

### 1. Vẫn lem vào nền / vá ngoài tay
- Tăng `feather` của node đó lên **25-30** (viền càng mờ càng blend tốt)
- Giảm `bbox_dilation` xuống **8** để BB hẹp lại
- Giảm `denoise` xuống **0.25** (chỉ làm mịn chứ không vẽ lại)

### 2. Tay/chân không được sửa
- Giảm `bbox_threshold` xuống **0.40–0.45** (nhạy hơn với các vật nhỏ)
- Tăng `bbox_dilation` lên **15**
- Đặt `cycle = 2` (quét 2 lượt)

### 3. Tay vẫn 6 ngón/biến dạng dù đã cắt đúng vùng
- Tăng `denoise` lên **0.40**, `steps` lên 22
- Thêm vào prompt tay: `perfect fingers, correct number of fingers`
- Thêm vào negative tay: `extra fingers, fewer fingers, mutated fingers`

### 4. Sau khi sửa tay/chan thấy viền bị đậm (đường chữ nhật)
- Tăng `feather` lên 25-30
- Tắt `noise_mask` (set `False`) hoặc ngược lại bật nó — tùy ảnh sẽ khác
- Giảm `denoise` 0.05

### 5. OOM VRAM
- Cell 3: `VRAM=lowvram`, `RESERVE_VRAM_GB=0.8`
- Tạm tắt 1 trong 3 detailer (thường bỏ chân nếu không thấy chân)

---

## Lưu ý chung

- **Tay luôn là phần khó nhất** của SD — không có giải pháp nào đúng 100% ngay lần đầu. Hãy sẵn sàng generate lại 2-3 seed.
- `hand_yolov8s.pt` (s=small) nhanh hơn và thường nhạy hơn `hand_yolov8m.pt` với các tay nhỏ/che khuất trong anime.
- Nếu ảnh nhiều người / tay chồng chéo → giảm `bbox_threshold` xuống 0.4 để không bỏ sót, nhưng tăng `drop_size` lên 50 để YOLO không bấm nhầm nền.
- Bạn luôn có thể bật lại SAM cho tay/chân (nối dây `sam_model_opt` vào node 30/40) nếu muốn thử — giữ `sam_detection_hint = center-1` và `sam_dilation_threshold = 0.93`.

Chúc ảnh ra đẹp 🎨
