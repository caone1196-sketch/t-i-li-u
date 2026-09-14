# -*- coding: utf-8 -*-
"""
GHÉP KHUNG + CHỮ TAROT ĐỒNG NHẤT
Cách dùng:  python tarot_ghep_khung.py <ảnh_tranh> "<TÊN LÁ BÀI>" "<SỐ La Mã>" <file_ra>
Ví dụ:      python tarot_ghep_khung.py art.png "THE SUN" "XIX" the_sun.png

Mọi lá bài ra cùng kích thước 750x1275 (tỉ lệ tarot chuẩn 2.75x4.75 inch),
cùng khung, cùng font, cùng vị trí chữ -> đồng nhất 100%.
"""
import sys
from PIL import Image, ImageDraw, ImageFont

# ================== TÙY CHỈNH KHUNG (sửa 1 lần, áp dụng cho cả bộ) ==================
CARD_W, CARD_H = 750, 1275          # kích thước lá bài (px)
MARGIN        = 45                   # viền ngoài đến khung
BORDER_COLOR  = (212, 175, 55)       # màu khung: vàng đồng cổ
BORDER_W      = 6                    # độ dày nét khung chính
INNER_GAP     = 12                   # khoảng cách khung kép
BG_COLOR      = (24, 18, 43)         # nền: xanh tím than huyền bí
TEXT_COLOR    = (212, 175, 55)       # màu chữ: đồng bộ khung
BANNER_H      = 110                  # chiều cao dải tên dưới
NUMBER_H      = 80                   # chiều cao ô số trên
# =====================================================================================

def find_font(size):
    """Tìm font serif có sẵn trên hệ thống (Colab/Linux/Windows)."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
        "C:/Windows/Fonts/georgiab.ttf",
        "C:/Windows/Fonts/timesbd.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()

def lam_la_bai(art_path, card_name, roman_numeral, out_path):
    # 1) Nền
    card = Image.new("RGB", (CARD_W, CARD_H), BG_COLOR)
    draw = ImageDraw.Draw(card)

    # 2) Vùng tranh (giữa số trên và tên dưới)
    art_x0 = MARGIN + BORDER_W + INNER_GAP
    art_y0 = MARGIN + NUMBER_H
    art_x1 = CARD_W - art_x0
    art_y1 = CARD_H - MARGIN - BANNER_H
    art_w, art_h = art_x1 - art_x0, art_y1 - art_y0

    # 3) Nạp tranh, cắt giữa cho khớp tỉ lệ (center-crop) rồi dán
    art = Image.open(art_path).convert("RGB")
    scale = max(art_w / art.width, art_h / art.height)
    new_size = (int(art.width * scale) + 1, int(art.height * scale) + 1)
    art = art.resize(new_size, Image.LANCZOS)
    left = (art.width - art_w) // 2
    top  = (art.height - art_h) // 2
    art = art.crop((left, top, left + art_w, top + art_h))
    card.paste(art, (art_x0, art_y0))

    # 4) Khung kép cổ điển
    draw.rectangle([MARGIN, MARGIN, CARD_W - MARGIN, CARD_H - MARGIN],
                   outline=BORDER_COLOR, width=BORDER_W)
    g = MARGIN + BORDER_W + INNER_GAP // 2
    draw.rectangle([g, g, CARD_W - g, CARD_H - g],
                   outline=BORDER_COLOR, width=2)
    # đường phân cách vùng tranh
    draw.line([art_x0, art_y0, art_x1, art_y0], fill=BORDER_COLOR, width=2)
    draw.line([art_x0, art_y1, art_x1, art_y1], fill=BORDER_COLOR, width=2)

    # 5) Số La Mã trên đỉnh
    font_num = find_font(44)
    bbox = draw.textbbox((0, 0), roman_numeral, font=font_num)
    tw = bbox[2] - bbox[0]
    draw.text(((CARD_W - tw) / 2, MARGIN + (NUMBER_H - 44) / 2 - 4),
              roman_numeral, fill=TEXT_COLOR, font=font_num)

    # 6) Tên lá bài dưới đáy (tự co chữ nếu tên dài)
    size = 56
    font_name = find_font(size)
    max_w = CARD_W - 2 * (MARGIN + BORDER_W + 20)
    while size > 24:
        bbox = draw.textbbox((0, 0), card_name, font=font_name)
        if bbox[2] - bbox[0] <= max_w:
            break
        size -= 4
        font_name = find_font(size)
    bbox = draw.textbbox((0, 0), card_name, font=font_name)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ty = art_y1 + (CARD_H - MARGIN - art_y1 - th) / 2 - bbox[1]
    draw.text(((CARD_W - tw) / 2, ty), card_name, fill=TEXT_COLOR, font=font_name)

    card.save(out_path)
    print(f"✅ Đã tạo: {out_path}  ({CARD_W}x{CARD_H})")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(1)
    lam_la_bai(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
