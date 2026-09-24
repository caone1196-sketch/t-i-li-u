# -*- coding: utf-8 -*-
"""
🎨 GIAO DIỆN FLUX Q5_K_S REALISTIC - Tối ưu người thật
Chạy trên Colab sau khi đã chạy Cell 3 (ComfyUI ở cổng 8188)

Tính năng:
- Txt2Img người thật với Flux Schnell Q5_K_S
- Prompt tự nhiên, không tag anime
- Tự động FaceDetailer + HandDetailer với SAM (người thật)
- Hires x1.5 tùy chọn
"""

import json, random, time, uuid, urllib.request, urllib.parse, io, os
COMFY = "http://127.0.0.1:8188"
CLIENT_ID = str(uuid.uuid4())

# Prompt mẫu người thật
PROMPT_MAU = {
    "Chân dung tự nhiên": "close-up portrait of a beautiful young Vietnamese woman, 24 years old, natural skin texture with visible pores, soft natural makeup, long straight black hair, soft window light, photorealistic, ultra detailed face, sharp eyes, DSLR, 85mm lens, bokeh background",
    "Full body ngoài trời": "full body photo of a young woman, 25 years old, athletic natural body, wearing elegant white summer dress, standing in a blooming garden, soft morning sunlight, natural pose, hands visible, five fingers, photorealistic, sharp focus, DSLR, 8k, detailed skin",
    "Fashion studio": "fashion model, 26 years old, flawless natural skin, professional makeup, studio softbox lighting, wearing black evening dress, elegant pose, photorealistic, ultra high resolution, detailed skin texture, sharp focus, 85mm lens",
    "Công sở": "beautiful office lady, 28 years old, wearing white shirt and black skirt, standing in modern office, natural smile, photorealistic, detailed face, natural skin, soft lighting, DSLR",
    "Cafe / đời thường": "young woman sitting in a cozy cafe, wearing casual sweater, holding coffee cup, natural candid moment, soft afternoon light, photorealistic, detailed skin, cozy atmosphere, DSLR photo, bokeh",
    "Áo dài Việt Nam": "beautiful Vietnamese woman wearing traditional white Ao Dai, standing in Hanoi old street, soft natural light, photorealistic, ultra detailed face, natural skin texture, elegant pose, DSLR, 85mm lens, bokeh",
}

NEG_MAC_DINH = "cartoon, anime, illustration, painting, 3d render, blurry, low quality, deformed, extra fingers, mutated hands, bad anatomy, watermark, text, oversaturated, plastic skin, airbrushed, doll, fake skin, lowres"

KICH_THUOC = {
    "Dọc 832x1216 (khuyên dùng)": (832, 1216),
    "Dọc cao 768x1344 (full body)": (768, 1344),
    "Vuông 1024x1024 (bán thân)": (1024, 1024),
    "Ngang 1216x832 (cinematic)": (1216, 832),
    "Siêu nét 1024x1536 (2K sau hires)": (1024, 1536),
}

def http_json(url, data=None):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())

def build_workflow_realistic(pos, neg, w, h, seed, hires=False):
    # Base workflow giống file workflow_flux_schnell_Q5_realistic.json
    wf = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "flux1-schnell-Q5_K_S.gguf"}},
        "2": {"class_type": "DualCLIPLoaderGGUF", "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5-v1_1-xxl-encoder-Q4_K_M.gguf", "type": "flux"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["2", 0]}},
        "11": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["2", 0]}},
        "20": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "21": {"class_type": "KSampler", "inputs": {"seed": seed, "control_after_generate": "randomize", "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0, "model": ["1",0], "positive": ["10",0], "negative": ["11",0], "latent_image": ["20",0]}},
        "30": {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": "bbox/face_yolov8m.pt"}},
        "31": {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": "bbox/hand_yolov8s.pt"}},
        "33": {"class_type": "SAMLoader", "inputs": {"model_name": "sam_vit_b_01ec64.pth", "device_mode": "AUTO"}},
        "34": {"class_type": "CLIPTextEncode", "inputs": {"text": "photorealistic, ultra detailed, sharp focus, natural skin texture, visible pores, same person, same composition, DSLR, 85mm lens", "clip": ["2",0]}},
        "35": {"class_type": "CLIPTextEncode", "inputs": {"text": "cartoon, anime, blurry, plastic skin, deformed", "clip": ["2",0]}},
    }
    if hires:
        wf["23"] = {"class_type": "LatentUpscale", "inputs": {"samples": ["21",0], "upscale_method": "bicubic", "width": int(w*1.5), "height": int(h*1.5), "crop": "disabled"}}
        wf["24"] = {"class_type": "KSampler", "inputs": {"seed": seed, "control_after_generate": "randomize", "steps": 6, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 0.35, "model": ["1",0], "positive": ["10",0], "negative": ["11",0], "latent_image": ["23",0]}}
        wf["22"] = {"class_type": "VAEDecode", "inputs": {"samples": ["24",0], "vae": ["3",0]}}
        max_size = max(int(w*1.5), int(h*1.5))
    else:
        wf["22"] = {"class_type": "VAEDecode", "inputs": {"samples": ["21",0], "vae": ["3",0]}}
        max_size = max(w,h)

    wf["40"] = {"class_type": "FaceDetailer", "inputs": {
        "image": ["22",0], "model": ["1",0], "clip": ["2",0], "vae": ["3",0],
        "positive": ["34",0], "negative": ["35",0],
        "bbox_detector": ["30",0], "sam_model_opt": ["33",0],
        "guide_size": 1024, "guide_size_for": True, "max_size": max_size,
        "seed": seed, "control_after_generate": "randomize", "steps": 8, "cfg": 1.0,
        "sampler_name": "euler", "scheduler": "simple", "denoise": 0.15,
        "feather": 5, "noise_mask": True, "force_inpaint": True,
        "bbox_threshold": 0.5, "bbox_dilation": 10, "bbox_crop_factor": 3.0,
        "sam_detection_hint": "center-1", "sam_threshold": 0.93, "sam_dilation": 0,
        "sam_bbox_expansion": 0, "sam_mask_hint_threshold": 0.7, "sam_mask_hint_use_negative": "Outter",
        "drop_size": 10, "wildcard": "detailed skin texture, natural pores, sharp eyes, detailed iris, detailed eyelashes, natural eyebrows, photorealistic skin",
        "cycle": 1, "inpaint_model": False, "noise_mask_feather": 5, "tiled_encode": True, "tiled_decode": True
    }}
    wf["41"] = {"class_type": "FaceDetailer", "inputs": {
        "image": ["40",0], "model": ["1",0], "clip": ["2",0], "vae": ["3",0],
        "positive": ["34",0], "negative": ["35",0],
        "bbox_detector": ["31",0], "sam_model_opt": ["33",0],
        "guide_size": 512, "guide_size_for": True, "max_size": max_size,
        "seed": seed, "control_after_generate": "randomize", "steps": 8, "cfg": 1.0,
        "sampler_name": "euler", "scheduler": "simple", "denoise": 0.22,
        "feather": 5, "noise_mask": True, "force_inpaint": True,
        "bbox_threshold": 0.35, "bbox_dilation": 10, "bbox_crop_factor": 2.5,
        "sam_detection_hint": "center-1", "sam_threshold": 0.93, "sam_dilation": 0,
        "sam_bbox_expansion": 0, "sam_mask_hint_threshold": 0.7, "sam_mask_hint_use_negative": "Outter",
        "drop_size": 10, "wildcard": "five fingers, perfect hands, detailed fingernails, natural knuckles, realistic skin texture, photorealistic hands",
        "cycle": 1, "inpaint_model": False, "noise_mask_feather": 5, "tiled_encode": True, "tiled_decode": True
    }}
    wf["50"] = {"class_type": "SaveImage", "inputs": {"filename_prefix": "FLUX_Realistic_Q5", "images": ["41",0]}}
    return wf

def tai_anh(prompt_id):
    from PIL import Image
    hist = http_json(f"{COMFY}/history/{prompt_id}")
    if prompt_id not in hist: return None
    for out in hist[prompt_id]["outputs"].values():
        for img in out.get("images", []):
            q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img.get("subfolder",""), "type": img.get("type","output")})
            with urllib.request.urlopen(f"{COMFY}/view?{q}", timeout=60) as r:
                return Image.open(io.BytesIO(r.read())).copy()
    return None

def tao_anh(prompt, mau_ten, neg_them, kt_ten, hires, seed_in):
    if not prompt.strip() and not mau_ten:
        yield None, "❌ Gõ prompt hoặc chọn mẫu"
        return
    if mau_ten and mau_ten in PROMPT_MAU:
        # Nếu user chọn mẫu mà ô prompt trống thì dùng mẫu, nếu có gõ thêm thì nối
        base = PROMPT_MAU[mau_ten]
        prompt_full = base if not prompt.strip() else f"{base}, {prompt.strip()}"
    else:
        prompt_full = prompt.strip()

    neg_full = NEG_MAC_DINH
    if neg_them.strip():
        neg_full += ", " + neg_them.strip()

    w,h = KICH_THUOC[kt_ten]
    seed = random.randint(0, 2**32) if int(seed_in) < 0 else int(seed_in)
    wf = build_workflow_realistic(prompt_full, neg_full, w, h, seed, hires=hires)

    yield None, f"⏳ Đang gửi workflow... {w}x{h} hires={hires} seed={seed}"
    try:
        res = http_json(f"{COMFY}/prompt", {"prompt": wf, "client_id": CLIENT_ID})
        prompt_id = res["prompt_id"]
    except Exception as e:
        yield None, f"❌ Không gửi được tới ComfyUI: {e}\n→ Kiểm tra Cell 3 đã chạy chưa"
        return

    # Đợi
    start = time.time()
    try:
        import websocket
        ws = websocket.create_connection(f"ws://127.0.0.1:8188/ws?clientId={CLIENT_ID}", timeout=5)
        ws.settimeout(2)
    except:
        ws = None

    done=False
    while not done and time.time()-start < 400:
        if ws:
            try:
                msg = ws.recv()
                if isinstance(msg,str):
                    m=json.loads(msg)
                    if m.get("type")=="progress":
                        d=m["data"]; pct=int(d["value"]/max(d["max"],1)*100)
                        bar="█"*(pct//5)+"░"*(20-pct//5)
                        yield None, f"🖌️ Đang vẽ {bar} {d['value']}/{d['max']} ({pct}%)"
                    elif m.get("type")=="executing" and m["data"].get("node") is None and m["data"].get("prompt_id")==prompt_id:
                        done=True
            except: pass
        else:
            time.sleep(2)
        try:
            hist=http_json(f"{COMFY}/history/{prompt_id}")
            if prompt_id in hist: done=True
        except: pass

    if ws:
        try: ws.close()
        except: pass

    img=None
    for _ in range(15):
        img=tai_anh(prompt_id)
        if img: break
        time.sleep(1)

    if not img:
        yield None, "❌ Hết giờ / không lấy được ảnh, xem log ComfyUI Cell 4"
        return
    elapsed=int(time.time()-start)
    yield img, f"✅ Xong sau {elapsed}s | Seed: {seed} | {w}x{h} hires={hires}\nPrompt: {prompt_full[:200]}..."

def main():
    import gradio as gr
    with gr.Blocks(title="FLUX Q5_K_S Realistic", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📸 FLUX Schnell Q5_K_S — Người thật Photorealistic\n**Model:** `flux1-schnell-Q5_K_S.gguf` + T5 Q4_K_M + SAM + YOLO face/hand\n**Workflow:** `workflow_flux_schnell_Q5_realistic.json` — tối ưu da thật, mắt nét, tay 5 ngón")
        with gr.Row():
            with gr.Column(scale=5):
                mau = gr.Radio(list(PROMPT_MAU.keys()), value="Chân dung tự nhiên", label="🎭 Mẫu prompt người thật")
                prompt = gr.Textbox(label="✏️ Prompt bổ sung (tiếng Anh, câu tự nhiên)", value="", lines=3, placeholder="VD: wearing red dress, smiling, standing near window...")
                with gr.Accordion("⚙️ Nâng cao", open=False):
                    kt = gr.Radio(list(KICH_THUOC.keys()), value="Dọc 832x1216 (khuyên dùng)", label="Kích thước")
                    hires = gr.Checkbox(value=False, label="🔍 Bật Hires x1.5 (1248x1824) — nét hơn nhưng chậm hơn 40%")
                    neg = gr.Textbox(label="Negative thêm", value="", lines=2)
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                btn = gr.Button("📸 TẠO ẢNH NGƯỜI THẬT", variant="primary", size="lg")
            with gr.Column(scale=5):
                anh = gr.Image(label="Kết quả", height=640, type="pil", format="png")
                log = gr.Textbox(label="Log", lines=4, value="Sẵn sàng. Chọn mẫu + bấm tạo ảnh.")
        btn.click(tao_anh, inputs=[prompt, mau, neg, kt, hires, seed], outputs=[anh, log])
        gr.Markdown("### 💡 Tip prompt người thật:\n- Thêm `DSLR, 85mm lens, bokeh, natural skin texture, visible pores`\n- Tránh `anime, cartoon, illustration`\n- Luôn thêm `detailed hands, five fingers` nếu thấy tay\n- Nếu da nhựa: thêm `natural skin texture, pores` + giảm denoise mặt xuống 0.12")
    demo.queue().launch(share=True, server_name="0.0.0.0", server_port=7860)

if __name__=="__main__":
    main()
