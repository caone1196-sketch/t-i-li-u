# -*- coding: utf-8 -*-
"""
🎨 GIAO DIỆN TẠO ẢNH GỌN GÀNG (tiếng Việt) — chạy đè lên ComfyUI

ComfyUI phải đang chạy nền ở cổng 8188 (Cell 3 trong notebook đã lo việc này).
Cách chạy trên Colab: xem Cell 3B trong ComfyUI_Colab_Free.ipynb
"""
import json
import random
import time
import uuid
import urllib.request

COMFY = "http://127.0.0.1:8188"
CLIENT_ID = str(uuid.uuid4())

QUALITY = "masterpiece, best quality, newest, absurdres, highres"

NEG_MAC_DINH = (
    "censored, mosaic censoring, blur censor, bar censor, pointless censoring, "
    "light censor, steam censor, convenient censoring, hair censor, novelty censor, "
    "worst quality, old, early, low quality, lowres, blurry, signature, username, logo, "
    "bad hands, mutated hands, extra digits, fewer digits, bad feet, "
    "mammal, anthro, furry, ambiguous form, feral, semi-anthro"
)

# Phong cách: tự thêm tag vào trước/sau prompt — bạn chỉ cần gõ NỘI DUNG ảnh
PRESET = {
    "Anime chuẩn": {
        "truoc": "",
        "sau": "",
        "neg": "",
    },
    "2.5D bán thực (giống thật)": {
        "truoc": "realistic, (photorealistic:1.1), ",
        "sau": ", detailed skin, detailed face, glossy hair, soft lighting, sharp focus",
        "neg": ", flat color, cel shading, chibi, sketch, doll",
    },
    "Lá bài Tarot (art nouveau)": {
        "truoc": "",
        "sau": (", art nouveau style, rich gold colors, ornate symbolic details, "
                "mystical atmosphere, vertical composition, detailed tarot deck illustration"),
        "neg": ", text, border",
    },
}

KICH_THUOC = {
    "Dọc 832×1216 (khuyên dùng)": (832, 1216),
    "Ngang 1216×832": (1216, 832),
    "Vuông 1024×1024": (1024, 1024),
}

SAMPLER = {
    "euler_ancestral (mặc định)": ("euler_ancestral", "normal"),
    "dpmpp_2m + karras (nét ổn định)": ("dpmpp_2m", "karras"),
}


def http_json(url, data=None):
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def lay_danh_sach_model():
    try:
        info = http_json(f"{COMFY}/object_info/CheckpointLoaderSimple")
        return list(info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0])
    except Exception:
        return []


def build_workflow(model, pos, neg, w, h, steps, cfg, seed, sampler, scheduler):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": w, "height": h, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["4", 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"images": ["6", 0], "filename_prefix": "GiaoDien"}},
    }


def tai_anh_ket_qua(prompt_id):
    """Đọc history -> tải ảnh PNG về dạng PIL Image."""
    from PIL import Image
    import io
    hist = http_json(f"{COMFY}/history/{prompt_id}")
    if prompt_id not in hist:
        return None
    for node_out in hist[prompt_id]["outputs"].values():
        for img in node_out.get("images", []):
            q = urllib.parse.urlencode(
                {"filename": img["filename"],
                 "subfolder": img.get("subfolder", ""),
                 "type": img.get("type", "output")})
            with urllib.request.urlopen(f"{COMFY}/view?{q}", timeout=60) as r:
                return Image.open(io.BytesIO(r.read())).copy()
    return None


import urllib.parse  # noqa: E402  (dùng trong tai_anh_ket_qua)


def tao_anh(model, preset_ten, prompt, neg_them, kt_ten, sampler_ten, steps, cfg, seed_nhap):
    """Generator: yield (ảnh, dòng trạng thái) để giao diện cập nhật dần."""
    if not model:
        yield None, "❌ Chưa có model! Kiểm tra Cell 2 đã tải model chưa, rồi chạy lại Cell 3 và 3B."
        return
    if not prompt.strip():
        yield None, "❌ Ô mô tả ảnh đang trống — gõ vài tag đã rồi bấm Vẽ nhé."
        return

    p = PRESET[preset_ten]
    prompt_full = p["truoc"] + prompt.strip().rstrip(",")
    if p["sau"]:
        prompt_full += p["sau"]
    if "masterpiece" not in prompt_full:
        prompt_full += ", " + QUALITY

    neg_full = NEG_MAC_DINH + p["neg"]
    if neg_them.strip():
        neg_full += ", " + neg_them.strip().strip(",")

    w, h = KICH_THUOC[kt_ten]
    sampler, scheduler = SAMPLER[sampler_ten]
    seed = random.randint(0, 2**48) if int(seed_nhap) < 0 else int(seed_nhap)

    wf = build_workflow(model, prompt_full, neg_full, w, h,
                        int(steps), float(cfg), seed, sampler, scheduler)

    # Kết nối WebSocket để nhận tiến trình từng bước (nếu được)
    ws = None
    try:
        import websocket
        ws = websocket.create_connection(
            f"ws://127.0.0.1:8188/ws?clientId={CLIENT_ID}", timeout=5)
        ws.settimeout(2)
    except Exception:
        ws = None

    yield None, "⏳ Đã gửi yêu cầu — đang xếp hàng..."
    try:
        res = http_json(f"{COMFY}/prompt", {"prompt": wf, "client_id": CLIENT_ID})
        prompt_id = res["prompt_id"]
    except Exception as e:
        yield None, f"❌ Không gửi được tới ComfyUI: {e}\n→ Chạy lại Cell 3 rồi chạy lại Cell 3B."
        return

    bat_dau = time.time()
    xong = False
    while not xong and time.time() - bat_dau < 600:
        if ws is not None:
            try:
                msg = ws.recv()
                if isinstance(msg, str):
                    m = json.loads(msg)
                    if m.get("type") == "progress":
                        d = m["data"]
                        pct = int(d["value"] / max(d["max"], 1) * 100)
                        thanh = "█" * (pct // 5) + "░" * (20 - pct // 5)
                        yield None, f"🖌️ Đang vẽ  {thanh}  bước {d['value']}/{d['max']} ({pct}%)"
                    elif (m.get("type") == "executing"
                          and m["data"].get("node") is None
                          and m["data"].get("prompt_id") == prompt_id):
                        xong = True
            except Exception:
                pass  # timeout 2s -> kiểm tra history bên dưới
        else:
            time.sleep(1.5)
        # Kiểm tra history (đường dự phòng khi không có WebSocket)
        if not xong:
            try:
                hist = http_json(f"{COMFY}/history/{prompt_id}")
                if prompt_id in hist:
                    xong = True
            except Exception:
                pass

    if ws is not None:
        try:
            ws.close()
        except Exception:
            pass

    anh = None
    for _ in range(10):  # ảnh có thể cần 1-2 giây để ghi xong
        anh = tai_anh_ket_qua(prompt_id)
        if anh is not None:
            break
        time.sleep(1)

    if anh is None:
        yield None, "❌ Hết giờ / không lấy được ảnh. Chạy Cell 4 xem log ComfyUI."
        return

    giay = int(time.time() - bat_dau)
    yield anh, (f"✅ Xong sau {giay} giây!   🌱 Seed: {seed}\n"
                f"(Lưu seed này lại nếu muốn vẽ lại đúng ảnh này — nhập vào ô Seed)")


def main():
    import gradio as gr

    models = lay_danh_sach_model()
    if not models:
        print("⚠️ Không kết nối được ComfyUI ở cổng 8188 — hãy chạy Cell 3 trước!")
    mac_dinh = next((m for m in models if "WAI" in m), models[0] if models else None)

    with gr.Blocks(title="Tạo ảnh AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎨 Tạo ảnh AI — bản gọn\n"
                    "Gõ **nội dung ảnh** bằng tag tiếng Anh (kiểu Danbooru), chọn phong cách, bấm Vẽ. "
                    "Tag chất lượng + tag loại trừ chống mờ đã được **tự thêm sẵn**.")
        with gr.Row():
            with gr.Column(scale=5):
                prompt = gr.Textbox(
                    label="✏️ Mô tả ảnh (chỉ cần nội dung — không cần tag chất lượng)",
                    value="1girl, long hair, school uniform, cherry blossoms, smile, upper body",
                    lines=3)
                preset = gr.Radio(list(PRESET.keys()), value="Anime chuẩn",
                                  label="🎭 Phong cách")
                kich_thuoc = gr.Radio(list(KICH_THUOC.keys()),
                                      value="Dọc 832×1216 (khuyên dùng)",
                                      label="📐 Kích thước")
                nut_ve = gr.Button("🖌️ VẼ ẢNH", variant="primary", size="lg")
                with gr.Accordion("⚙️ Nâng cao (thường không cần đụng)", open=False):
                    model = gr.Dropdown(models, value=mac_dinh, label="Model")
                    sampler = gr.Radio(list(SAMPLER.keys()),
                                       value="euler_ancestral (mặc định)", label="Sampler")
                    steps = gr.Slider(15, 40, value=27, step=1, label="Steps")
                    cfg = gr.Slider(3, 8, value=5, step=0.5, label="CFG")
                    seed = gr.Number(value=-1, precision=0,
                                     label="Seed (-1 = ngẫu nhiên; nhập số cũ để vẽ lại y hệt)")
                    neg_them = gr.Textbox(
                        label="Loại trừ THÊM (bộ chống mờ đã có sẵn, chỉ điền nếu muốn thêm)",
                        value="", lines=2)
            with gr.Column(scale=5):
                anh = gr.Image(label="🖼️ Kết quả (chuột phải → Save image để tải về)",
                               height=620)
                trang_thai = gr.Textbox(label="Tiến trình", value="Sẵn sàng.", lines=2)

        nut_ve.click(tao_anh,
                     inputs=[model, preset, prompt, neg_them, kich_thuoc,
                             sampler, steps, cfg, seed],
                     outputs=[anh, trang_thai])

    demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
