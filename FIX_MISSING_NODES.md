# 🔧 FIX LỖI "2 nodes affected - Missing node type"

Bạn gặp lỗi này khi load workflow `flux1-schnell-Q5_K_S.gguf`:

```
**2 nodes affected**
Missing node type
Missing Node Packs
```

## Nguyên nhân

Workflow GGUF dùng 2 node custom:
- `UnetLoaderGGUF` 
- `DualCLIPLoaderGGUF`

2 node này nằm trong pack **ComfyUI-GGUF** (city96). Nếu bạn chưa cài pack này, ComfyUI sẽ báo đỏ 2 node.

## Cách fix (3 cách)

### Cách 1: Cài ComfyUI-GGUF (khuyên dùng - để dùng Q5_K_S)

**Trong Colab, chạy lại Cell 1B mới nhất** (đã fix):

Cell 1B mới sẽ tự:
```bash
git clone https://github.com/city96/ComfyUI-GGUF.git
```

Sau đó **restart Cell 3** (khởi động lại ComfyUI) để nó nhận node mới.

Hoặc cài tay trong ComfyUI Manager:
1. Mở ComfyUI → Click **Manager** (bên phải)
2. **Custom Nodes Manager** → Search `GGUF`
3. Cài `ComfyUI-GGUF` → Restart ComfyUI

### Cách 2: Dùng workflow FP8 không cần GGUF (fix nhanh nhất)

Nếu bạn không muốn cài GGUF, dùng bản FP8 - **KHÔNG CẦN cài thêm gì**, chỉ cần Impact Pack (đã có sẵn):

- `workflow_flux_schnell_fp8_realistic_minimal.json` - 8 node, chỉ built-in, chạy 100%
- `workflow_flux_schnell_fp8_realistic_v2_best.json` - 15 node, cần Impact Pack (FaceDetailer)

Bản FP8 chất lượng gần bằng Q5_K_S (FP8 ~11GB, Q5 ~8.3GB + T5 2.9GB = 12GB), chạy nhanh hơn.

**Model cần cho FP8:**
- `flux1-schnell-fp8.safetensors` (11GB) trong `models/checkpoints/`
- `clip_l.safetensors` + `t5xxl_fp8_e4m3fn.safetensors` trong `models/clip/`
- `ae.safetensors` trong `models/vae/`

Cell 2 mới đã tự tải FP8 nếu bạn tick BO_QUA_MODEL=False.

### Cách 3: Cài Manager trước rồi cài GGUF

Nếu ComfyUI báo `pip install -U --pre comfyui-manager`:

Trong Colab chạy:
```bash
!pip install -U --pre comfyui-manager
```

Hoặc trong Cell 1B:
```bash
!cd /content/ComfyUI/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Manager.git
```

Sau đó restart Cell 3 với flag `--enable-manager` (Cell 3 mới đã tự thêm).

## Kiểm tra đã cài thành công chưa

Sau khi cài xong, trong log ComfyUI (`/content/comfyui.log`) phải thấy:
```
[ComfyUI-GGUF] - Using llama.cpp...
```

Hoặc trong ComfyUI → Manager → Installed → thấy `ComfyUI-GGUF`

## Workflow nào dùng khi nào?

| Workflow | Cần cài gì | Khi nào dùng |
|---|---|---|
| `*_minimal.json` (8 node) | Không cần gì, built-in | Test nhanh, fix lỗi |
| `*_fp8_*_minimal.json` | Không cần GGUF | Muốn chạy ngay không cài thêm |
| `*_fp8_*_v2_best.json` | Impact Pack (đã có) | Muốn người thật đẹp, không muốn cài GGUF |
| `*_Q5_*_minimal.json` | ComfyUI-GGUF | Test GGUF |
| `*_Q5_*_v2_best.json` | ComfyUI-GGUF + Impact Pack | **Tốt nhất, khuyên dùng** - chất lượng cao nhất dưới 15GB |

## Vẫn lỗi?

1. Chạy Cell 4 kiểm tra: `!ls /content/ComfyUI/custom_nodes | grep -i gguf`
   - Nếu không thấy `ComfyUI-GGUF` → Cell 1B chưa chạy thành công
2. Chạy Cell 4: `!tail -50 /content/comfyui.log | grep -i gguf`
   - Nếu thấy lỗi `No module named gguf` → `!pip install gguf`
3. Restart hoàn toàn: Runtime → Restart runtime → chạy lại Cell 1 → 1B → 2 → 3

