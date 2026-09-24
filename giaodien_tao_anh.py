# -*- coding: utf-8 -*-
"""
🎨 GIAO DIỆN TẠO ẢNH & SỬA ẢNH NUDE — NoobAI-XL 1.1 (V-Pred) — chạy đè lên ComfyUI

ComfyUI phải đang chạy nền ở cổng 8188 (Cell 3 trong notebook đã lo việc này).
Cách chạy trên Colab: xem Cell 3B trong ComfyUI_Colab_Free.ipynb

Model mặc định: noobai-XL-1.1.safetensors (file Drive của bạn) — Euler + normal, CFG 4-5, steps 28-35.

Quy trình:
- Tab 1: Tạo ảnh nude từ text (txt2img)
- Tab 2: Sửa ảnh có sẵn thành nude (img2img / undress) — upload ảnh mặc đồ → đổi thành nude
- Tab 3: Inpaint — chỉ xóa vùng chọn (vẽ mask lên áo/quần)
"""
import json
import random
import time
import uuid
import urllib.request
import urllib.parse
import os
import io

COMFY = "http://127.0.0.1:8188"
CLIENT_ID = str(uuid.uuid4())

QUALITY = "masterpiece, best quality, newest, absurdres, highres"
QUALITY_NUDE = "masterpiece, best quality, newest, absurdres, highres, extremely detailed, detailed skin, detailed body"

NEG_MAC_DINH = (
    "censored, mosaic censoring, blur censor, bar censor, pointless censoring, "
    "light censor, steam censor, convenient censoring, hair censor, novelty censor, "
    "worst quality, old, early, low quality, lowres, blurry, signature, username, logo, "
    "bad hands, mutated hands, extra digits, fewer digits, bad feet, "
    "mammal, anthro, furry, ambiguous form, feral, semi-anthro"
)

NEG_NUDE = NEG_MAC_DINH + ", clothes, dress, shirt, bikini, bra, panties"

# Phong cách
PRESET = {
    "Anime chuẩn": {"truoc": "", "sau": "", "neg": ""},
    "2.5D bán thực": {
        "truoc": "realistic, (photorealistic:1.1), ",
        "sau": ", detailed skin, detailed face, glossy hair, soft lighting, sharp focus",
        "neg": ", flat color, cel shading, chibi, sketch, doll",
    },
    "Nude - đứng": {"truoc": "", "sau": ", standing, nude, naked, nsfw, fully nude, no clothes", "neg": ""},
    "Nude - nằm": {"truoc": "", "sau": ", lying on bed, nude, naked, nsfw, fully nude, spread legs, bedroom, soft lighting", "neg": ""},
    "Nude - ngồi": {"truoc": "", "sau": ", sitting, nude, naked, nsfw, large breasts, detailed body, chair", "neg": ""},
}

KICH_THUOC = {
    "Dọc 832×1216 (khuyên dùng)": (832, 1216),
    "Ngang 1216×832": (1216, 832),
    "Vuông 1024×1024": (1024, 1024),
}

SAMPLER = {
    "euler + normal (NoobAI V-Pred — mặc định)": ("euler", "normal"),
    "euler_ancestral (WAI)": ("euler_ancestral", "normal"),
    "dpmpp_2m + karras (WAI, nét ổn định)": ("dpmpp_2m", "karras"),
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
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["4", 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "NoobAI_Nude_Txt2Img"}},
    }

def build_workflow_img2img(model, pos, neg, steps, cfg, seed, sampler, scheduler, denoise, image_filename):
    # LoadImage -> VAEEncode -> KSampler -> VAEDecode
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["1", 2]}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["5", 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": sampler, "scheduler": scheduler, "denoise": denoise}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "NoobAI_Nude_Img2Img"}},
    }

def build_workflow_inpaint(model, pos, neg, steps, cfg, seed, sampler, scheduler, denoise, image_filename, mask_filename):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "5": {"class_type": "LoadImage", "inputs": {"image": mask_filename}},
        "6": {"class_type": "ImageToMask", "inputs": {"image": ["5", 0], "channel": "red"}},
        "7": {"class_type": "VAEEncodeForInpaint", "inputs": {"pixels": ["4", 0], "mask": ["6", 0], "vae": ["1", 2], "grow_mask_by": 6}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["7", 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": sampler, "scheduler": scheduler, "denoise": denoise}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "NoobAI_Nude_Inpaint"}},
    }

def upload_image_to_comfy(pil_image, filename="upload.png"):
    """Upload PIL Image lên ComfyUI /upload/image, trả về tên file trên server"""
    # ComfyUI upload expects multipart/form-data
    import requests
    # Fallback nếu không có requests thì dùng urllib
    try:
        # thử requests
        buf = io.BytesIO()
        pil_image.save(buf, format='PNG')
        buf.seek(0)
        files = {'image': (filename, buf, 'image/png')}
        data = {'overwrite': 'true'}
        resp = requests.post(f"{COMFY}/upload/image", files=files, data=data, timeout=60)
        resp.raise_for_status()
        j = resp.json()
        return j['name']
    except Exception as e:
        # fallback urllib
        try:
            import http.client
            import mimetypes
            buf = io.BytesIO()
            pil_image.save(buf, format='PNG')
            image_data = buf.getvalue()
            boundary = '----WebKitFormBoundary' + uuid.uuid4().hex[:16]
            body = b''
            # overwrite field
            body += f'--{boundary}\r\n'.encode()
            body += b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'
            body += f'--{boundary}\r\n'.encode()
            body += f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode()
            body += b'Content-Type: image/png\r\n\r\n'
            body += image_data + b'\r\n'
            body += f'--{boundary}--\r\n'.encode()
            req = urllib.request.Request(
                f"{COMFY}/upload/image",
                data=body,
                headers={'Content-Type': f'multipart/form-data; boundary={boundary}'}
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                j = json.loads(r.read())
                return j['name']
        except Exception as e2:
            raise RuntimeError(f"Upload ảnh lên ComfyUI thất bại: {e} / {e2}")

def tai_anh_ket_qua(prompt_id):
    from PIL import Image
    hist = http_json(f"{COMFY}/history/{prompt_id}")
    if prompt_id not in hist:
        return None
    for node_out in hist[prompt_id]["outputs"].values():
        for img in node_out.get("images", []):
            q = urllib.parse.urlencode(
                {"filename": img["filename"], "subfolder": img.get("subfolder", ""), "type": img.get("type", "output")})
            with urllib.request.urlopen(f"{COMFY}/view?{q}", timeout=60) as r:
                return Image.open(io.BytesIO(r.read())).copy()
    return None

def run_workflow_with_progress(wf):
    """Gửi workflow và đợi, yield trạng thái"""
    ws = None
    try:
        import websocket
        ws = websocket.create_connection(f"ws://127.0.0.1:8188/ws?clientId={CLIENT_ID}", timeout=5)
        ws.settimeout(2)
    except Exception:
        ws = None

    try:
        res = http_json(f"{COMFY}/prompt", {"prompt": wf, "client_id": CLIENT_ID})
        prompt_id = res["prompt_id"]
    except Exception as e:
        yield None, f"❌ Không gửi được tới ComfyUI: {e}\n→ Chạy lại Cell 3 rồi chạy lại Cell 3B."
        return None

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
                    elif (m.get("type") == "executing" and m["data"].get("node") is None and m["data"].get("prompt_id") == prompt_id):
                        xong = True
            except Exception:
                pass
        else:
            time.sleep(1.5)
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
    for _ in range(10):
        anh = tai_anh_ket_qua(prompt_id)
        if anh is not None:
            break
        time.sleep(1)

    return anh, prompt_id, time.time() - bat_dau

def tao_anh(model, preset_ten, prompt, neg_them, kt_ten, sampler_ten, steps, cfg, seed_nhap):
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
        prompt_full += ", " + QUALITY_NUDE

    neg_full = NEG_MAC_DINH + p["neg"]
    if "nude" in prompt_full.lower() or "naked" in prompt_full.lower() or "nsfw" in prompt_full.lower():
        # nếu là nude thì thêm NEG_NUDE để tránh che
        neg_full = NEG_NUDE + p["neg"]
    if neg_them.strip():
        neg_full += ", " + neg_them.strip().strip(",")

    w, h = KICH_THUOC[kt_ten]
    sampler, scheduler = SAMPLER[sampler_ten]
    seed = random.randint(0, 2**48) if int(seed_nhap) < 0 else int(seed_nhap)

    wf = build_workflow(model, prompt_full, neg_full, w, h, int(steps), float(cfg), seed, sampler, scheduler)

    yield None, "⏳ Đã gửi yêu cầu — đang xếp hàng..."
    result = yield from run_workflow_with_progress_gen(wf)
    if result is None:
        return
    anh, prompt_id, elapsed = result
    if anh is None:
        yield None, "❌ Hết giờ / không lấy được ảnh. Chạy Cell 4 xem log ComfyUI."
        return
    giay = int(elapsed)
    yield anh, f"✅ Xong sau {giay} giây!   🌱 Seed: {seed}\n(Lưu seed này lại nếu muốn vẽ lại đúng ảnh này — nhập vào ô Seed)"

def run_workflow_with_progress_gen(wf):
    """Wrapper để dùng yield from trong tao_anh"""
    ws = None
    try:
        import websocket
        ws = websocket.create_connection(f"ws://127.0.0.1:8188/ws?clientId={CLIENT_ID}", timeout=5)
        ws.settimeout(2)
    except Exception:
        ws = None

    try:
        res = http_json(f"{COMFY}/prompt", {"prompt": wf, "client_id": CLIENT_ID})
        prompt_id = res["prompt_id"]
    except Exception as e:
        yield None, f"❌ Không gửi được tới ComfyUI: {e}\n→ Chạy lại Cell 3 rồi chạy lại Cell 3B."
        return None

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
                    elif (m.get("type") == "executing" and m["data"].get("node") is None and m["data"].get("prompt_id") == prompt_id):
                        xong = True
            except Exception:
                pass
        else:
            time.sleep(1.5)
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
    for _ in range(10):
        anh = tai_anh_ket_qua(prompt_id)
        if anh is not None:
            break
        time.sleep(1)

    if anh is None:
        yield None, "❌ Hết giờ / không lấy được ảnh"
        return None

    return anh, prompt_id, time.time() - bat_dau

def tao_anh_img2img(model, prompt, neg_them, input_image, denoise, sampler_ten, steps, cfg, seed_nhap):
    if not model:
        yield None, "❌ Chưa có model!"
        return
    if input_image is None:
        yield None, "❌ Chưa upload ảnh gốc — hãy upload ảnh mặc đồ cần đổi thành nude"
        return
    if not prompt.strip():
        prompt = "nude, naked, nsfw, fully nude, no clothes, detailed skin, large breasts, perfect body"

    prompt_full = QUALITY_NUDE + ", " + prompt.strip().rstrip(",")
    neg_full = NEG_NUDE
    if neg_them.strip():
        neg_full += ", " + neg_them.strip().strip(",")

    sampler, scheduler = SAMPLER[sampler_ten]
    seed = random.randint(0, 2**48) if int(seed_nhap) < 0 else int(seed_nhap)

    # Upload ảnh lên ComfyUI
    yield None, "⏳ Đang upload ảnh lên ComfyUI..."
    try:
        from PIL import Image
        if isinstance(input_image, dict):  # gradio may return dict
            pil = Image.open(input_image['name'])
        else:
            pil = input_image
        # Đảm bảo RGB
        if pil.mode != 'RGB':
            pil = pil.convert('RGB')
        filename = f"upload_{uuid.uuid4().hex[:8]}.png"
        server_filename = upload_image_to_comfy(pil, filename)
    except Exception as e:
        yield None, f"❌ Upload ảnh thất bại: {e}"
        return

    wf = build_workflow_img2img(model, prompt_full, neg_full, int(steps), float(cfg), seed, sampler, scheduler, float(denoise), server_filename)

    yield None, f"⏳ Đã upload {server_filename} — đang xử lý img2img denoise={denoise}..."
    # chạy workflow
    ws = None
    try:
        import websocket
        ws = websocket.create_connection(f"ws://127.0.0.1:8188/ws?clientId={CLIENT_ID}", timeout=5)
        ws.settimeout(2)
    except Exception:
        ws = None

    try:
        res = http_json(f"{COMFY}/prompt", {"prompt": wf, "client_id": CLIENT_ID})
        prompt_id = res["prompt_id"]
    except Exception as e:
        yield None, f"❌ Không gửi được tới ComfyUI: {e}"
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
                        yield None, f"🖌️ Đang sửa ảnh  {thanh}  bước {d['value']}/{d['max']} ({pct}%)"
                    elif (m.get("type") == "executing" and m["data"].get("node") is None and m["data"].get("prompt_id") == prompt_id):
                        xong = True
            except Exception:
                pass
        else:
            time.sleep(1.5)
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
    for _ in range(10):
        anh = tai_anh_ket_qua(prompt_id)
        if anh is not None:
            break
        time.sleep(1)

    if anh is None:
        yield None, "❌ Hết giờ / không lấy được ảnh"
        return

    giay = int(time.time() - bat_dau)
    yield anh, f"✅ Xong sau {giay} giây! Denoise={denoise} — Seed: {seed}\nMẹo: denoise 0.55-0.65 giữ dáng, 0.7-0.85 đổi nhiều hơn"

def tao_anh_inpaint(model, prompt, neg_them, input_image, mask_image, denoise, sampler_ten, steps, cfg, seed_nhap):
    if not model:
        yield None, "❌ Chưa có model!"
        return
    if input_image is None:
        yield None, "❌ Chưa upload ảnh gốc"
        return
    if mask_image is None:
        yield None, "❌ Chưa có mask — vẽ vùng trắng cần đổi thành nude trong tab Mask (hoặc upload ảnh mask trắng/đen)"
        return
    if not prompt.strip():
        prompt = "nude, naked, nsfw, bare breasts, bare pussy, fully nude, detailed skin, no clothes"

    prompt_full = QUALITY_NUDE + ", " + prompt.strip().rstrip(",")
    neg_full = NEG_NUDE
    if neg_them.strip():
        neg_full += ", " + neg_them.strip().strip(",")

    sampler, scheduler = SAMPLER[sampler_ten]
    seed = random.randint(0, 2**48) if int(seed_nhap) < 0 else int(seed_nhap)

    yield None, "⏳ Upload ảnh + mask..."
    try:
        from PIL import Image
        # input
        pil_in = input_image
        if isinstance(pil_in, dict):
            pil_in = Image.open(pil_in['name'])
        if pil_in.mode != 'RGB':
            pil_in = pil_in.convert('RGB')
        # mask - gradio ImageMask có thể trả về dict với mask
        pil_mask = None
        if isinstance(mask_image, dict):
            # gradio sketch returns {'image': ..., 'mask': ...}
            if 'mask' in mask_image and mask_image['mask'] is not None:
                pil_mask = mask_image['mask']
                if isinstance(pil_mask, dict):
                    pil_mask = Image.open(pil_mask['name'])
            elif 'image' in mask_image:
                pil_mask = mask_image['image']
                if isinstance(pil_mask, dict):
                    pil_mask = Image.open(pil_mask['name'])
                else:
                    pil_mask = pil_mask
            else:
                pil_mask = Image.open(mask_image['name']) if 'name' in mask_image else None
        else:
            pil_mask = mask_image

        # Nếu mask là RGBA, lấy alpha hoặc chuyển sang L
        if pil_mask is None:
            raise RuntimeError("Không đọc được mask")

        if hasattr(pil_mask, 'mode') and pil_mask.mode == 'RGBA':
            # Lấy alpha làm mask
            pil_mask = pil_mask.split()[-1]
        if pil_mask.mode != 'RGB':
            # Chuyển sang RGB để upload (ImageToMask sẽ lấy kênh red)
            if pil_mask.mode == 'L':
                pil_mask = pil_mask.convert('RGB')
            else:
                pil_mask = pil_mask.convert('RGB')

        fn_in = f"inpaint_in_{uuid.uuid4().hex[:8]}.png"
        fn_mask = f"inpaint_mask_{uuid.uuid4().hex[:8]}.png"
        server_in = upload_image_to_comfy(pil_in, fn_in)
        server_mask = upload_image_to_comfy(pil_mask, fn_mask)
    except Exception as e:
        yield None, f"❌ Upload thất bại: {e}"
        return

    wf = build_workflow_inpaint(model, prompt_full, neg_full, int(steps), float(cfg), seed, sampler, scheduler, float(denoise), server_in, server_mask)

    yield None, f"⏳ Đã upload — đang inpaint denoise={denoise}..."
    # chạy
    ws = None
    try:
        import websocket
        ws = websocket.create_connection(f"ws://127.0.0.1:8188/ws?clientId={CLIENT_ID}", timeout=5)
        ws.settimeout(2)
    except Exception:
        ws = None

    try:
        res = http_json(f"{COMFY}/prompt", {"prompt": wf, "client_id": CLIENT_ID})
        prompt_id = res["prompt_id"]
    except Exception as e:
        yield None, f"❌ Không gửi được: {e}"
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
                        yield None, f"🖌️ Inpaint  {thanh}  {d['value']}/{d['max']} ({pct}%)"
                    elif (m.get("type") == "executing" and m["data"].get("node") is None and m["data"].get("prompt_id") == prompt_id):
                        xong = True
            except Exception:
                pass
        else:
            time.sleep(1.5)
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
    for _ in range(10):
        anh = tai_anh_ket_qua(prompt_id)
        if anh is not None:
            break
        time.sleep(1)

    if anh is None:
        yield None, "❌ Hết giờ"
        return

    giay = int(time.time() - bat_dau)
    yield anh, f"✅ Inpaint xong sau {giay}s — Seed: {seed}"

def main():
    import gradio as gr
    import os

    models = lay_danh_sach_model()
    if not models:
        print("⚠️ Không kết nối được ComfyUI ở cổng 8188 — hãy chạy Cell 3 trước!")

    drive_model = ""
    try:
        if os.path.isfile('/content/model_file.txt'):
            drive_model = open('/content/model_file.txt', encoding='utf-8', errors='ignore').read().strip()
    except:
        pass
    mac_dinh = None
    if drive_model:
        mac_dinh = next((m for m in models if m == drive_model or drive_model.lower() in m.lower() or m.lower() in drive_model.lower()), None)
    if not mac_dinh:
        mac_dinh = next((m for m in models if "noobai" in m.lower()),
                        next((m for m in models if "wai" in m.lower()),
                             models[0] if models else None))

    with gr.Blocks(title="Tạo ảnh & Sửa ảnh Nude - NoobAI", theme=gr.themes.Soft()) as demo:
        md = "# 🎨 Tạo ảnh & Sửa ảnh Nude — NoobAI-XL 1.1 (V-Pred)\n**Model:** `noobai-XL-1.1.safetensors` — Euler + normal, CFG 4-5, Steps 28-35. Kéo workflow JSON trong repo nếu muốn dùng ComfyUI thuần."
        if drive_model:
            md += f"\n\n✅ **Model từ Drive của bạn:** `{drive_model}` — đã tự chọn."
        gr.Markdown(md)

        with gr.Tab("1️⃣ Tạo ảnh nude từ text (Txt2Img)"):
            with gr.Row():
                with gr.Column(scale=5):
                    prompt = gr.Textbox(label="✏️ Mô tả ảnh (tag Danbooru, tiếng Anh)", value="1girl, solo, long hair, nude, naked, nsfw, large breasts, sitting, bedroom, soft lighting", lines=3)
                    preset = gr.Radio(list(PRESET.keys()), value="Nude - ngồi", label="🎭 Phong cách nhanh")
                    kich_thuoc = gr.Radio(list(KICH_THUOC.keys()), value="Dọc 832×1216 (khuyên dùng)", label="📐 Kích thước")
                    nut_ve = gr.Button("🖌️ VẼ ẢNH NUDE", variant="primary", size="lg")
                    with gr.Accordion("⚙️ Nâng cao", open=False):
                        model = gr.Dropdown(models, value=mac_dinh, label="Model")
                        sampler = gr.Radio(list(SAMPLER.keys()), value="euler + normal (NoobAI V-Pred — mặc định)", label="Sampler")
                        steps = gr.Slider(10, 50, value=30, step=1, label="Steps (NoobAI: 28–35)")
                        cfg = gr.Slider(1, 10, value=4.5, step=0.5, label="CFG (NoobAI: 4–5)")
                        seed = gr.Number(value=-1, precision=0, label="Seed (-1 = ngẫu nhiên)")
                        neg_them = gr.Textbox(label="Loại trừ thêm (đã có sẵn censored, clothes...)", value="", lines=2)
                with gr.Column(scale=5):
                    anh = gr.Image(label="🖼️ Kết quả", height=620, format="png", type="pil")
                    trang_thai = gr.Textbox(label="Tiến trình", value="Sẵn sàng.", lines=3)
            nut_ve.click(tao_anh, inputs=[model, preset, prompt, neg_them, kich_thuoc, sampler, steps, cfg, seed], outputs=[anh, trang_thai])

        with gr.Tab("2️⃣ Sửa ảnh có sẵn thành nude (Img2Img / Undress)"):
            gr.Markdown("Upload ảnh mặc đồ → AI sẽ đổi thành nude. Denoise thấp giữ dáng, cao đổi nhiều.")
            with gr.Row():
                with gr.Column(scale=5):
                    input_img = gr.Image(label="📤 Ảnh gốc (mặc đồ)", type="pil", height=400)
                    prompt2 = gr.Textbox(label="Prompt nude muốn đổi", value="nude, naked, nsfw, fully nude, no clothes, large breasts, detailed skin, perfect body", lines=2)
                    denoise = gr.Slider(0.3, 0.9, value=0.65, step=0.05, label="Denoise (0.55-0.65 giữ dáng, 0.7-0.85 sáng tạo)")
                    nut_ve2 = gr.Button("🔄 ĐỔI THÀNH NUDE", variant="primary", size="lg")
                    with gr.Accordion("⚙️ Nâng cao", open=False):
                        model2 = gr.Dropdown(models, value=mac_dinh, label="Model")
                        sampler2 = gr.Radio(list(SAMPLER.keys()), value="euler + normal (NoobAI V-Pred — mặc định)", label="Sampler")
                        steps2 = gr.Slider(10, 50, value=30, step=1, label="Steps")
                        cfg2 = gr.Slider(1, 10, value=4.5, step=0.5, label="CFG")
                        seed2 = gr.Number(value=-1, precision=0, label="Seed")
                        neg2 = gr.Textbox(label="Negative thêm", value="", lines=2)
                with gr.Column(scale=5):
                    anh2 = gr.Image(label="🖼️ Kết quả nude", height=620, format="png", type="pil")
                    trang_thai2 = gr.Textbox(label="Tiến trình", value="Sẵn sàng.", lines=3)
            nut_ve2.click(tao_anh_img2img, inputs=[model2, prompt2, neg2, input_img, denoise, sampler2, steps2, cfg2, seed2], outputs=[anh2, trang_thai2])

        with gr.Tab("3️⃣ Inpaint - Xóa áo/quần (chính xác)"):
            gr.Markdown("Vẽ mask trắng lên vùng cần xóa (áo, quần) → chỉ vùng đó thành nude. Dùng brush vẽ trong ô Mask.")
            with gr.Row():
                with gr.Column(scale=5):
                    input_img3 = gr.Image(label="📤 Ảnh gốc", type="pil", height=350)
                    # Gradio ImageMask: cho phép vẽ mask
                    mask_img = gr.ImageMask(label="🎨 Vẽ mask trắng lên vùng cần đổi thành nude (áo/quần)", type="pil", height=350)
                    prompt3 = gr.Textbox(label="Prompt vùng inpaint", value="nude, bare breasts, bare pussy, fully nude, detailed skin, no clothes", lines=2)
                    denoise3 = gr.Slider(0.5, 1.0, value=0.95, step=0.05, label="Denoise (inpaint nên 0.9-1.0)")
                    nut_ve3 = gr.Button("✂️ INPAINT NUDE", variant="primary", size="lg")
                    with gr.Accordion("⚙️ Nâng cao", open=False):
                        model3 = gr.Dropdown(models, value=mac_dinh, label="Model")
                        sampler3 = gr.Radio(list(SAMPLER.keys()), value="euler + normal (NoobAI V-Pred — mặc định)", label="Sampler")
                        steps3 = gr.Slider(10, 50, value=32, step=1, label="Steps")
                        cfg3 = gr.Slider(1, 10, value=4.5, step=0.5, label="CFG")
                        seed3 = gr.Number(value=-1, precision=0, label="Seed")
                        neg3 = gr.Textbox(label="Negative thêm", value="", lines=2)
                with gr.Column(scale=5):
                    anh3 = gr.Image(label="🖼️ Kết quả inpaint", height=620, format="png", type="pil")
                    trang_thai3 = gr.Textbox(label="Tiến trình", value="Sẵn sàng.", lines=3)
            nut_ve3.click(tao_anh_inpaint, inputs=[model3, prompt3, neg3, input_img3, mask_img, denoise3, sampler3, steps3, cfg3, seed3], outputs=[anh3, trang_thai3])

        with gr.Tab("📂 Workflow JSON"):
            gr.Markdown("Kéo các file JSON này vào ComfyUI để dùng workflow thuần (không cần Gradio):\n- `workflow_noobai_nude_txt2img.json`: tạo nude từ text\n- `workflow_noobai_nude_img2img.json`: sửa ảnh mặc đồ thành nude\n- `workflow_noobai_nude_inpaint.json`: inpaint xóa áo/quần\n- `workflow_noobai_nude_facedetail.json`: tự làm đẹp mặt sau khi nude\n\nFile đã có sẵn trong repo, tải về từ tab Files bên trái.")

    demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
