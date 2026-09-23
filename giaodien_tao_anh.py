# -*- coding: utf-8 -*-
"""
🎨 GIAO DIỆN TẠO ẢNH & SỬA ẢNH NUDE — NoobAI-XL 1.1 (V-Pred) — chạy đè lên ComfyUI

ComfyUI phải đang chạy nền ở cổng 8188 (Cell 3 trong notebook đã lo việc này).
Cách chạy trên Colab: xem Cell 3B trong ComfyUI_Colab_Free.ipynb

Model mặc định: noobai-XL-1.1.safetensors (file Drive của bạn) — Euler + normal, CFG 4-5, steps 28-35.

Quy trình:
- Tab 1: Tạo ảnh nude từ text (txt2img)
- Tab 2: Sửa ảnh có sẵn thành nude (img2img / undress) — upload ảnh mặc đồ → đổi thành nude
- Tab 3: Sửa tay / chân / mặt / Inpaint vùng tô đen chuyên dụng (hỗ trợ tự bắt vùng đen hoặc vẽ mask)
- Tab 4: Inpaint xóa áo/quần (cũ)
- Tab 5: Workflow JSON
"""
import json
import random
import time
import uuid
import urllib.request
import urllib.parse
import os
import io
import numpy as np
from PIL import Image, ImageFilter

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

# Preset sửa tay, chân, mặt, anatomy
PRESET_INPAINT_FIX = {
    "🖐️ Sửa tay (Fix Hands)": {
        "prompt": "perfect anime hands, detailed fingers, 5 fingers, beautiful slender hands, anatomically correct hands, delicate fingers, highly detailed skin",
        "neg": "bad hands, mutated hands, extra digits, fewer digits, missing fingers, fused fingers, distorted fingers, extra fingers, malformed limbs",
        "denoise": 0.85,
    },
    "🦶 Sửa chân / ngón chân (Fix Feet)": {
        "prompt": "beautiful anime feet, perfect toes, 5 toes, slender ankles, detailed soles, anatomically correct feet, smooth legs, delicate feet",
        "neg": "bad feet, mutated feet, extra toes, fewer toes, missing toes, fused toes, deformed limbs, distorted feet",
        "denoise": 0.85,
    },
    "👁️ Sửa khuôn mặt & mắt (Fix Face & Eyes)": {
        "prompt": "masterpiece, beautiful detailed face, symmetrical detailed eyes, expressive eyes, cute anime mouth, sharp focus, perfect anime face, soft skin, blush",
        "neg": "ugly face, deformed eyes, crossed eyes, bad pupils, blurry face, distorted mouth, bad teeth, asymmetry",
        "denoise": 0.75,
    },
    "👙 Đổi quần áo thành da Nude (Undress Inpaint)": {
        "prompt": "nude, naked, nsfw, bare breasts, bare nipples, bare pussy, fully nude, smooth skin, detailed body, no clothes, natural lighting",
        "neg": "clothes, dress, shirt, bra, panties, swimwear, underwear, censor, mosaic",
        "denoise": 0.95,
    },
    "✨ Vẽ lại tự do (Custom Prompt)": {
        "prompt": "masterpiece, best quality, detailed, natural skin texture, seamless blend",
        "neg": "worst quality, low quality, blurry, deformed, artifact, glitch",
        "denoise": 0.85,
    },
}

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
    "Dọc 832×1216 (Chuẩn Anime khuyên dùng)": (832, 1216),
    "Dọc 896×1152 (Tỉ lệ 3:4 chân dung)": (896, 1152),
    "Dọc 768×1344 (Tỉ lệ 9:16 Story/Điện thoại)": (768, 1344),
    "Dọc 704×1408 (Toàn thân siêu dài)": (704, 1408),
    "Vuông 1024×1024 (Tỉ lệ 1:1 Avatar)": (1024, 1024),
    "Ngang 1216×832 (Ngang khuyên dùng)": (1216, 832),
    "Ngang 1152×896 (Tỉ lệ 4:3 phong cảnh)": (1152, 896),
    "Ngang 1344×768 (Tỉ lệ 16:9 Hình nền máy tính)": (1344, 768),
    "Ngang 1536×640 (Tỉ lệ 21:9 Siêu rộng Cinematic)": (1536, 640),
    "⚙️ Tùy chỉnh (Nhập Width / Height tự do)": (0, 0),
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

def build_workflow_inpaint(model, pos, neg, steps, cfg, seed, sampler, scheduler, denoise, image_filename, mask_filename, grow_mask=6):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": model}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_filename}},
        "5": {"class_type": "LoadImage", "inputs": {"image": mask_filename}},
        "6": {"class_type": "ImageToMask", "inputs": {"image": ["5", 0], "channel": "red"}},
        "7": {"class_type": "VAEEncodeForInpaint", "inputs": {"pixels": ["4", 0], "mask": ["6", 0], "vae": ["1", 2], "grow_mask_by": int(grow_mask)}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["7", 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": sampler, "scheduler": scheduler, "denoise": denoise}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["1", 2]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "NoobAI_Nude_Inpaint"}},
    }

def upload_image_to_comfy(pil_image, filename="upload.png"):
    """Upload PIL Image lên ComfyUI /upload/image, trả về tên file trên server"""
    try:
        import requests
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
        try:
            buf = io.BytesIO()
            pil_image.save(buf, format='PNG')
            image_data = buf.getvalue()
            boundary = '----WebKitFormBoundary' + uuid.uuid4().hex[:16]
            body = b''
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

def run_workflow_with_progress_gen(wf):
    """Wrapper nhận workflow và yield thanh tiến trình qua WebSocket"""
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

def tao_anh(model, preset_ten, prompt, neg_them, kt_ten, custom_w, custom_h, sampler_ten, steps, cfg, seed_nhap):
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
        neg_full = NEG_NUDE + p["neg"]
    if neg_them.strip():
        neg_full += ", " + neg_them.strip().strip(",")

    if kt_ten == "⚙️ Tùy chỉnh (Nhập Width / Height tự do)":
        w = int(custom_w) if custom_w and int(custom_w) > 64 else 832
        h = int(custom_h) if custom_h and int(custom_h) > 64 else 1216
    else:
        w, h = KICH_THUOC.get(kt_ten, (832, 1216))
    # Bo tròn kích thước về bội số của 64 hoặc 8 (tốt nhất cho SDXL)
    w = (w // 8) * 8
    h = (h // 8) * 8
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

    yield None, "⏳ Đang upload ảnh lên ComfyUI..."
    try:
        if isinstance(input_image, dict):
            pil = Image.open(input_image['name'])
        else:
            pil = input_image
        if pil.mode != 'RGB':
            pil = pil.convert('RGB')
        filename = f"upload_{uuid.uuid4().hex[:8]}.png"
        server_filename = upload_image_to_comfy(pil, filename)
    except Exception as e:
        yield None, f"❌ Upload ảnh thất bại: {e}"
        return

    wf = build_workflow_img2img(model, prompt_full, neg_full, int(steps), float(cfg), seed, sampler, scheduler, float(denoise), server_filename)

    yield None, f"⏳ Đã upload {server_filename} — đang xử lý img2img denoise={denoise}..."
    result = yield from run_workflow_with_progress_gen(wf)
    if result is None:
        return
    anh, prompt_id, elapsed = result
    if anh is None:
        yield None, "❌ Hết giờ / không lấy được ảnh"
        return
    giay = int(elapsed)
    yield anh, f"✅ Xong sau {giay} giây! Denoise={denoise} — Seed: {seed}\nMẹo: denoise 0.55-0.65 giữ dáng, 0.7-0.85 đổi nhiều hơn"

def parse_input_image_and_mask(source_input, mask_input=None, detect_black=True, black_threshold=35, mask_dilation=4, mask_blur=2):
    """
    Trích xuất ảnh gốc và mask (màu trắng trên nền đen).
    Hỗ trợ:
    1. Vùng vẽ đè (ImageMask / ImageEditor từ Gradio)
    2. Tự động nhận diện pixel màu đen (RGB <= black_threshold) nếu người dùng tô đen bằng bút / paint ngoài rồi upload
    """
    pil_in = None
    pil_mask = None

    if source_input is None:
        return None, None, "Chưa cung cấp ảnh đầu vào"

    # Xử lý input từ Gradio (dict / Image.Image / filepath)
    if isinstance(source_input, dict):
        bg = source_input.get('background')
        composite = source_input.get('composite')
        image = source_input.get('image')

        # Ưu tiên lấy background (ảnh gốc trước khi bị vẽ đè màu cọ)
        for candidate in [bg, composite, image]:
            if candidate is not None:
                if isinstance(candidate, Image.Image):
                    pil_in = candidate.copy()
                    break
                elif isinstance(candidate, str) and os.path.isfile(candidate):
                    pil_in = Image.open(candidate).copy()
                    break
                elif isinstance(candidate, dict) and 'name' in candidate:
                    pil_in = Image.open(candidate['name']).copy()
                    break

        # Trích xuất mask vẽ từ layers của Gradio ImageMask / ImageEditor
        layers = source_input.get('layers', [])
        if layers:
            combined_alpha = None
            for l in layers:
                if isinstance(l, Image.Image):
                    l_rgba = l.convert('RGBA')
                    l_alpha = l_rgba.split()[-1]
                    l_arr = np.array(l_alpha)
                    if np.count_nonzero(l_arr > 10) > 10:
                        if combined_alpha is None:
                            combined_alpha = l_arr
                        else:
                            combined_alpha = np.maximum(combined_alpha, l_arr)
            if combined_alpha is not None:
                pil_mask = Image.fromarray(combined_alpha, mode='L')

        # Trích xuất từ mask dict nếu có
        if pil_mask is None and 'mask' in source_input and source_input['mask'] is not None:
            m = source_input['mask']
            if isinstance(m, dict) and 'name' in m:
                m = Image.open(m['name'])
            if isinstance(m, Image.Image):
                if m.mode == 'RGBA':
                    pil_mask = m.split()[-1]
                else:
                    pil_mask = m.convert('L')
    elif isinstance(source_input, Image.Image):
        pil_in = source_input.copy()
    elif isinstance(source_input, str) and os.path.isfile(source_input):
        pil_in = Image.open(source_input).copy()

    # Nếu người dùng có upload mask riêng
    if mask_input is not None and pil_mask is None:
        if isinstance(mask_input, Image.Image):
            pil_mask = mask_input
        elif isinstance(mask_input, dict):
            if 'mask' in mask_input and mask_input['mask'] is not None:
                m = mask_input['mask']
                pil_mask = Image.open(m['name']) if isinstance(m, dict) and 'name' in m else m
            elif 'image' in mask_input and mask_input['image'] is not None:
                m = mask_input['image']
                pil_mask = Image.open(m['name']) if isinstance(m, dict) and 'name' in m else m

    if pil_in is not None:
        pil_in = pil_in.convert('RGB')

    if pil_mask is not None:
        if isinstance(pil_mask, Image.Image):
            if pil_mask.mode == 'RGBA':
                pil_mask = pil_mask.split()[-1]
            else:
                pil_mask = pil_mask.convert('L')

    msg_detail = []
    has_drawn_mask = (pil_mask is not None and np.count_nonzero(np.array(pil_mask) > 10) > 15)

    # Nhận diện vùng màu đen tô trên ảnh gốc
    if (detect_black or not has_drawn_mask) and pil_in is not None:
        arr = np.array(pil_in)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        # Pixel đen: R, G, B <= black_threshold
        black_pixels = (r <= black_threshold) & (g <= black_threshold) & (b <= black_threshold)
        black_count = np.count_nonzero(black_pixels)
        if black_count > 15:
            black_mask_arr = np.where(black_pixels, 255, 0).astype(np.uint8)
            detected_black_mask = Image.fromarray(black_mask_arr, mode='L')

            # Tẩy phần đen thành màu trung bình của vùng xung quanh để VAE không bị lem mực đen khi inpaint
            surrounding = (~black_pixels)
            if np.count_nonzero(surrounding) > 50:
                mean_color = arr[surrounding].mean(axis=0).astype(np.uint8)
                arr_cleaned = arr.copy()
                arr_cleaned[black_pixels] = mean_color
                pil_in = Image.fromarray(arr_cleaned, mode='RGB')

            if pil_mask is not None and has_drawn_mask:
                combined = np.maximum(np.array(pil_mask), np.array(detected_black_mask))
                pil_mask = Image.fromarray(combined, mode='L')
                msg_detail.append(f"Gộp mask cọ vẽ + {black_count} pixel vùng bôi đen (ngưỡng {black_threshold})")
            else:
                pil_mask = detected_black_mask
                msg_detail.append(f"Tự động nhận diện {black_count} pixel vùng bôi đen (ngưỡng {black_threshold})")
        elif not has_drawn_mask:
            # Nếu ngưỡng mặc định không ra, thử ngưỡng cao hơn một chút (ngưỡng 55)
            black_pixels_relaxed = (r <= 55) & (g <= 55) & (b <= 55)
            black_count_relaxed = np.count_nonzero(black_pixels_relaxed)
            if black_count_relaxed > 25:
                black_mask_arr = np.where(black_pixels_relaxed, 255, 0).astype(np.uint8)
                pil_mask = Image.fromarray(black_mask_arr, mode='L')
                surrounding = (~black_pixels_relaxed)
                if np.count_nonzero(surrounding) > 50:
                    mean_color = arr[surrounding].mean(axis=0).astype(np.uint8)
                    arr_cleaned = arr.copy()
                    arr_cleaned[black_pixels_relaxed] = mean_color
                    pil_in = Image.fromarray(arr_cleaned, mode='RGB')
                msg_detail.append(f"Tự động nhận diện {black_count_relaxed} pixel vùng bôi đen gần đúng (ngưỡng 55)")

    if pil_mask is None or np.count_nonzero(np.array(pil_mask) > 10) < 10:
        return pil_in, None, "❌ Chưa thấy mask vẽ hoặc vùng đen cần sửa. Hãy dùng bút vẽ lên ảnh hoặc tô đen vùng lỗi (tay/chân/mặt)!"

    # Khớp kích thước
    if pil_in is not None and pil_mask.size != pil_in.size:
        pil_mask = pil_mask.resize(pil_in.size, Image.Resampling.NEAREST)

    # Mở rộng biên (dilation)
    if mask_dilation > 0:
        filter_size = max(3, int(mask_dilation) * 2 + 1)
        pil_mask = pil_mask.filter(ImageFilter.MaxFilter(size=filter_size))

    # Làm mờ viền (feather/blur)
    if mask_blur > 0:
        pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(radius=int(mask_blur)))

    # Đưa về RGB (cho ComfyUI ImageToMask red channel)
    pil_mask = pil_mask.convert('RGB')
    status_str = " | ".join(msg_detail) if msg_detail else "Đã nhận diện mask thành công"
    return pil_in, pil_mask, status_str

def tao_anh_inpaint_sua_chi_tiet(
    model, preset_fix, prompt_inpaint, neg_them, input_image,
    denoise, sampler_ten, steps, cfg, seed_nhap,
    detect_black, black_threshold, mask_dilation, mask_blur
):
    """Hàm inpaint tối ưu chuyên sửa tay, chân, mặt, vùng tô đen"""
    if not model:
        yield None, None, "❌ Chưa có model! Hãy kiểm tra Cell 2."
        return
    if input_image is None:
        yield None, None, "❌ Chưa tải ảnh lên. Hãy upload ảnh cần sửa tay/chân/mặt."
        return

    yield None, None, "🔍 Đang phân tích ảnh & bóc tách vùng tô đen / mask vẽ..."
    pil_in, pil_mask, msg = parse_input_image_and_mask(
        input_image, None,
        detect_black=bool(detect_black),
        black_threshold=int(black_threshold),
        mask_dilation=int(mask_dilation),
        mask_blur=int(mask_blur)
    )

    if pil_mask is None:
        yield None, None, msg
        return

    # Prompt từ preset
    p_cfg = PRESET_INPAINT_FIX.get(preset_fix, PRESET_INPAINT_FIX["✨ Vẽ lại tự do (Custom Prompt)"])
    chosen_prompt = prompt_inpaint.strip() if prompt_inpaint.strip() else p_cfg["prompt"]
    prompt_full = QUALITY + ", " + chosen_prompt.rstrip(",")
    neg_full = NEG_MAC_DINH + ", " + p_cfg["neg"]
    if neg_them.strip():
        neg_full += ", " + neg_them.strip().strip(",")

    sampler, scheduler = SAMPLER[sampler_ten]
    seed = random.randint(0, 2**48) if int(seed_nhap) < 0 else int(seed_nhap)

    yield None, pil_mask, f"⏳ {msg} — Đang tải lên ComfyUI..."
    try:
        fn_in = f"fix_in_{uuid.uuid4().hex[:8]}.png"
        fn_mask = f"fix_mask_{uuid.uuid4().hex[:8]}.png"
        server_in = upload_image_to_comfy(pil_in, fn_in)
        server_mask = upload_image_to_comfy(pil_mask, fn_mask)
    except Exception as e:
        yield None, pil_mask, f"❌ Upload ảnh lên ComfyUI thất bại: {e}"
        return

    wf = build_workflow_inpaint(
        model, prompt_full, neg_full,
        int(steps), float(cfg), seed, sampler, scheduler,
        float(denoise), server_in, server_mask, grow_mask=mask_dilation
    )

    yield None, pil_mask, f"⏳ Đang inpaint sửa chi tiết (denoise={denoise}, steps={steps})..."
    
    # Chạy workflow và yield đúng 3 outputs cho Gradio
    gen = run_workflow_with_progress_gen(wf)
    result = None
    try:
        while True:
            val = next(gen)
            # val là (anh, status)
            yield val[0], pil_mask, val[1]
    except StopIteration as e:
        result = e.value

    if result is None:
        yield None, pil_mask, "❌ Quá trình chạy bị gián đoạn."
        return

    anh, prompt_id, elapsed = result
    if anh is None:
        yield None, pil_mask, "❌ Hết giờ / không lấy được ảnh từ ComfyUI."
        return
    giay = int(elapsed)
    yield anh, pil_mask, f"✅ Sửa xong sau {giay}s! ({preset_fix}) — Seed: {seed}\n{msg}\nMẹo: Nếu tay/chân vẫn méo, tăng Denoise lên 0.85-0.95 hoặc tô vùng đen rộng hơn một chút để AI có không gian vẽ lại hoàn chỉnh."

def tao_anh_inpaint(model, prompt, neg_them, input_image, mask_image, denoise, sampler_ten, steps, cfg, seed_nhap):
    if not model:
        yield None, "❌ Chưa có model!"
        return
    if input_image is None:
        yield None, "❌ Chưa upload ảnh gốc"
        return
    if not prompt.strip():
        prompt = "nude, naked, nsfw, bare breasts, bare pussy, fully nude, detailed skin, no clothes"

    prompt_full = QUALITY_NUDE + ", " + prompt.strip().rstrip(",")
    neg_full = NEG_NUDE
    if neg_them.strip():
        neg_full += ", " + neg_them.strip().strip(",")

    sampler, scheduler = SAMPLER[sampler_ten]
    seed = random.randint(0, 2**48) if int(seed_nhap) < 0 else int(seed_nhap)

    yield None, "⏳ Phân tích ảnh và mask..."
    pil_in, pil_mask, msg = parse_input_image_and_mask(
        input_image, mask_image,
        detect_black=True, black_threshold=40,
        mask_dilation=6, mask_blur=2
    )

    if pil_mask is None:
        yield None, msg
        return

    try:
        fn_in = f"inpaint_in_{uuid.uuid4().hex[:8]}.png"
        fn_mask = f"inpaint_mask_{uuid.uuid4().hex[:8]}.png"
        server_in = upload_image_to_comfy(pil_in, fn_in)
        server_mask = upload_image_to_comfy(pil_mask, fn_mask)
    except Exception as e:
        yield None, f"❌ Upload thất bại: {e}"
        return

    wf = build_workflow_inpaint(model, prompt_full, neg_full, int(steps), float(cfg), seed, sampler, scheduler, float(denoise), server_in, server_mask)

    yield None, f"⏳ Đã upload — đang inpaint denoise={denoise}..."
    result = yield from run_workflow_with_progress_gen(wf)
    if result is None:
        return
    anh, prompt_id, elapsed = result
    if anh is None:
        yield None, "❌ Hết giờ"
        return
    giay = int(elapsed)
    yield anh, f"✅ Inpaint xong sau {giay}s — Seed: {seed}\n{msg}"


def update_kich_thuoc_visibility(kt_ten):
    is_custom = (kt_ten == "⚙️ Tùy chỉnh (Nhập Width / Height tự do)")
    return gr.update(visible=is_custom)

def update_inpaint_preset_vals(preset_name):
    if preset_name in PRESET_INPAINT_FIX:
        p = PRESET_INPAINT_FIX[preset_name]
        return p["prompt"], p["denoise"]
    return "", 0.85

def main():
    import gradio as gr

    models = lay_danh_sach_model()
    if not models:
        print("⚠️ Không kết nối được ComfyUI ở cổng 8188 — hãy chạy Cell 3 trước!")

    drive_model = ""
    try:
        if os.path.isfile('/content/model_file.txt'):
            drive_model = open('/content/model_file.txt', encoding='utf-8', errors='ignore').read().strip()
    except Exception:
        pass
    mac_dinh = None
    if drive_model:
        mac_dinh = next((m for m in models if m == drive_model or drive_model.lower() in m.lower() or m.lower() in drive_model.lower()), None)
    if not mac_dinh:
        mac_dinh = next((m for m in models if "noobai" in m.lower()),
                        next((m for m in models if "wai" in m.lower()),
                             models[0] if models else None))

    custom_css = """
    .gradio-container { max-width: 1200px !important; margin: 0 auto !important; }
    .status-box textarea { font-family: monospace; }
    """

    with gr.Blocks(title="Tạo ảnh & Sửa Tay Chân Mặt Nude - NoobAI", theme=gr.themes.Soft(), css=custom_css) as demo:
        md = "# 🎨 Giao Diện Tạo & Sửa Ảnh Anime Nude — NoobAI-XL 1.1 (V-Pred)\n"
        md += "**Tối ưu 2026**: Hỗ trợ vẽ ảnh nude, sửa ảnh có sẵn, và **sửa tay/chân/mặt bị lỗi bằng cách tô đen hoặc vẽ mask**."
        if drive_model:
            md += f"\n\n📂 **Model Drive tự nhận diện:** `{drive_model}`"
        gr.Markdown(md)

        # TAB 1
        with gr.Tab("1️⃣ Tạo ảnh nude từ text (Txt2Img)"):
            with gr.Row():
                with gr.Column(scale=5):
                    prompt = gr.Textbox(label="✏️ Mô tả ảnh (tag Danbooru, tiếng Anh)", value="1girl, solo, long hair, nude, naked, nsfw, large breasts, sitting, bedroom, soft lighting", lines=3)
                    preset = gr.Radio(list(PRESET.keys()), value="Nude - ngồi", label="🎭 Phong cách nhanh")
                    kich_thuoc = gr.Dropdown(
                        list(KICH_THUOC.keys()),
                        value="Dọc 832×1216 (Chuẩn Anime khuyên dùng)",
                        label="📐 Tùy chọn Độ phân giải & Tỉ lệ khung hình (Aspect Ratio)"
                    )
                    with gr.Row(visible=False) as custom_res_row:
                        custom_w = gr.Slider(512, 2048, value=832, step=64, label="Chiều rộng (Width px)")
                        custom_h = gr.Slider(512, 2048, value=1216, step=64, label="Chiều cao (Height px)")
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
                    trang_thai = gr.Textbox(label="Tiến trình", value="Sẵn sàng.", lines=3, elem_classes=["status-box"])
            kich_thuoc.change(
                update_kich_thuoc_visibility,
                inputs=[kich_thuoc],
                outputs=[custom_res_row]
            )
            nut_ve.click(
                tao_anh,
                inputs=[model, preset, prompt, neg_them, kich_thuoc, custom_w, custom_h, sampler, steps, cfg, seed],
                outputs=[anh, trang_thai]
            )

        # TAB 2
        with gr.Tab("2️⃣ Sửa ảnh có sẵn thành nude (Img2Img / Undress)"):
            gr.Markdown("Upload ảnh mặc đồ → AI sẽ đổi thành nude toàn thân. Denoise thấp giữ dáng, cao đổi nhiều.")
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
                    trang_thai2 = gr.Textbox(label="Tiến trình", value="Sẵn sàng.", lines=3, elem_classes=["status-box"])
            nut_ve2.click(tao_anh_img2img, inputs=[model2, prompt2, neg2, input_img, denoise, sampler2, steps2, cfg2, seed2], outputs=[anh2, trang_thai2])

        # TAB 3 (MỚI: SỬA TAY / CHÂN / MẶT / VÙNG TÔ ĐEN CHUYÊN DỤNG)
        with gr.Tab("✨ Sửa tay / chân / mặt (Khu vực tô đen)"):
            gr.Markdown(
                "### 🛠️ Chức năng sửa lỗi Anatomy bằng vùng tô đen\n"
                "**Cách dùng tiện lợi:**\n"
                "1. **Cách 1 (Tô đen sẵn):** Lấy app vẽ / Paint tô màu đen đặc `(RGB #000000)` đè lên bàn tay, bàn chân, khuôn mặt bị méo rồi upload vào đây.\n"
                "2. **Cách 2 (Vẽ trực tiếp):** Upload ảnh và dùng công cụ cọ vẽ (brush) bôi trắng lên vùng cần vẽ lại.\n"
                "3. Hệ thống sẽ **tự động bóc tách vùng tô đen**, mở rộng viền (dilation) và vẽ lại chi tiết cực nét bằng NoobAI."
            )
            with gr.Row():
                with gr.Column(scale=5):
                    input_fix_img = gr.ImageMask(
                        label="📤 Upload ảnh đã tô đen HOẶC vẽ trực tiếp lên tay/chân/mặt",
                        type="pil",
                        height=420
                    )
                    preset_fix_radio = gr.Radio(
                        list(PRESET_INPAINT_FIX.keys()),
                        value="🖐️ Sửa tay (Fix Hands)",
                        label="🎯 Mục tiêu cần sửa"
                    )
                    prompt_fix = gr.Textbox(
                        label="Prompt chi tiết (được điền tự động theo mục tiêu)",
                        value=PRESET_INPAINT_FIX["🖐️ Sửa tay (Fix Hands)"]["prompt"],
                        lines=2
                    )
                    denoise_fix = gr.Slider(
                        0.5, 1.0,
                        value=PRESET_INPAINT_FIX["🖐️ Sửa tay (Fix Hands)"]["denoise"],
                        step=0.05,
                        label="Denoise (Tay/Chân nên 0.8–0.9, Mặt nên 0.7–0.8)"
                    )
                    nut_fix = gr.Button("✨ TIẾN HÀNH SỬA VÙNG TÔ ĐEN", variant="primary", size="lg")

                    with gr.Accordion("⚙️ Tùy chỉnh nhận diện vùng đen & viền mask", open=False):
                        detect_black_cb = gr.Checkbox(value=True, label="Tự động nhận diện pixel màu đen (Black detection)")
                        black_thresh = gr.Slider(5, 80, value=35, step=1, label="Ngưỡng đen RGB (mặc định 35: nhận diện các màu đen < 35/255)")
                        mask_dilation_sl = gr.Slider(0, 20, value=6, step=1, label="Mở rộng viền Mask (Dilation - giúp ghép mượt)")
                        mask_blur_sl = gr.Slider(0, 10, value=2, step=1, label="Làm mờ biên Mask (Blur/Feather)")
                        model_fix = gr.Dropdown(models, value=mac_dinh, label="Model")
                        sampler_fix = gr.Radio(list(SAMPLER.keys()), value="euler + normal (NoobAI V-Pred — mặc định)", label="Sampler")
                        steps_fix = gr.Slider(15, 50, value=32, step=1, label="Steps")
                        cfg_fix = gr.Slider(1, 10, value=4.5, step=0.5, label="CFG")
                        seed_fix = gr.Number(value=-1, precision=0, label="Seed (-1 = ngẫu nhiên)")
                        neg_fix_custom = gr.Textbox(label="Negative loại trừ thêm", value="", lines=1)

                with gr.Column(scale=5):
                    anh_fix_ketqua = gr.Image(label="🖼️ Kết quả sau khi sửa", height=420, format="png", type="pil")
                    with gr.Accordion("👁️ Xem vùng Mask hệ thống nhận diện", open=False):
                        mask_preview = gr.Image(label="Vùng Mask được bóc tách (trắng là vùng sửa)", height=220, type="pil")
                    trang_thai_fix = gr.Textbox(label="Tiến trình xử lý", value="Sẵn sàng.", lines=4, elem_classes=["status-box"])

            preset_fix_radio.change(
                update_inpaint_preset_vals,
                inputs=[preset_fix_radio],
                outputs=[prompt_fix, denoise_fix]
            )

            nut_fix.click(
                tao_anh_inpaint_sua_chi_tiet,
                inputs=[
                    model_fix, preset_fix_radio, prompt_fix, neg_fix_custom, input_fix_img,
                    denoise_fix, sampler_fix, steps_fix, cfg_fix, seed_fix,
                    detect_black_cb, black_thresh, mask_dilation_sl, mask_blur_sl
                ],
                outputs=[anh_fix_ketqua, mask_preview, trang_thai_fix]
            )

        # TAB 4: Inpaint xóa quần áo (tổng quát)
        with gr.Tab("4️⃣ Inpaint - Xóa áo/quần (cũ)"):
            gr.Markdown("Vẽ mask hoặc tô đen lên vùng cần xóa áo/quần để chuyển thành da nude.")
            with gr.Row():
                with gr.Column(scale=5):
                    input_img3 = gr.Image(label="📤 Ảnh gốc", type="pil", height=350)
                    mask_img = gr.ImageMask(label="🎨 Vẽ mask hoặc upload ảnh mask", type="pil", height=350)
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
                    trang_thai3 = gr.Textbox(label="Tiến trình", value="Sẵn sàng.", lines=3, elem_classes=["status-box"])
            nut_ve3.click(tao_anh_inpaint, inputs=[model3, prompt3, neg3, input_img3, mask_img, denoise3, sampler3, steps3, cfg3, seed3], outputs=[anh3, trang_thai3])

        # TAB 5: Workflow
        with gr.Tab("📂 Workflow JSON"):
            gr.Markdown("Kéo các file JSON này vào ComfyUI để dùng workflow thuần (không cần Gradio):\n- `workflow_noobai_nude_txt2img.json`: tạo nude từ text\n- `workflow_noobai_nude_img2img.json`: sửa ảnh mặc đồ thành nude\n- `workflow_noobai_nude_inpaint.json`: inpaint xóa áo/quần\n- `workflow_noobai_nude_facedetail.json`: tự làm đẹp mặt sau khi nude\n\nFile đã có sẵn trong repo, tải về từ tab Files bên trái.")

    demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
