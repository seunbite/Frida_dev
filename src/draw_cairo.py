# frida_pycairo.py  ── run:  python frida_pycairo.py
import math, cairo

W, H = 768, 1038               # 원본 해상도
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
ctx = cairo.Context(surface)

# ────────────────────────────── 팔레트
NAVY    = (0.04, 0.07, 0.10)   # 배경
TEAL    = (0.05, 0.26, 0.34)   # 물결
ROBES   = (0.42, 0.06, 0.16)
GOLD    = (0.85, 0.66, 0.25)
SCARLET = (0.73, 0.09, 0.11)
SKIN_LT = (0.97, 0.80, 0.69)   # 밝은 피부
SKIN_DK = (0.86, 0.63, 0.55)   # 음영
HAIR    = (0.09, 0.09, 0.09)

def rgb(rgb_tuple): ctx.set_source_rgb(*rgb_tuple)

# ───────────────────────────── 배경
rgb(NAVY); ctx.rectangle(0, 0, W, H); ctx.fill()

def swirl(cx, cy, r, loops=2):
    ctx.save(); ctx.translate(cx, cy)
    ctx.new_path(); ctx.move_to(r, 0)
    for i in range(loops * 4):
        ctx.rel_curve_to( 0, -r/2, -r, -r/2, -r, 0)
        ctx.rel_curve_to( 0,  r/2,  r,  r/2,  r, 0)
    ctx.restore(); ctx.stroke()

ctx.set_line_width(4); rgb(TEAL)
for x in range(-150, W+150, 150):
    swirl(x, 520, 60, loops=1)

# ───────────────────────────── 로브
rgb(ROBES)
ctx.rectangle(150, 560, 470, 390)
ctx.fill()

# 골드/적색 칼라
def collar(side):                 # side = -1(left) or 1(right)
    ctx.save()
    ctx.translate(W/2 + 35*side, 540)
    ctx.scale(side, 1)
    path = [(0, 0), ( 60, -320), ( 90, -400), (120, -450)]
    ctx.move_to(*path[0])
    for x,y in path[1:]:
        ctx.line_to(x, y)
    ctx.line_to(0, -30); ctx.close_path()
    ctx.restore()

for s in (-1, 1):
    collar(s); rgb(GOLD); ctx.fill_preserve(); ctx.set_line_width(12); ctx.stroke()
    collar(s); rgb(SCARLET); ctx.set_line_width(6); ctx.stroke()

# ───────────────────────────── 목 & 얼굴 (그라디언트)
neck = cairo.LinearGradient(W/2, 380, W/2, 620)
neck.add_color_stop_rgb(0, *SKIN_LT)
neck.add_color_stop_rgb(1, *SKIN_DK)
ctx.rectangle(W/2-45, 380, 90, 240); ctx.set_source(neck); ctx.fill()

head = cairo.RadialGradient(W/2, 300, 40, W/2, 300, 125)
head.add_color_stop_rgb(0, *SKIN_LT)
head.add_color_stop_rgb(1, *SKIN_DK)
ctx.arc(W/2, 300, 125, 0, 2*math.pi); ctx.set_source(head); ctx.fill()

# ───────────────────────────── 헤어 & 눈썹
rgb(HAIR)
ctx.arc(W/2, 280, 125, math.pi, 2*math.pi)  # 앞머리 반원
ctx.line_to(W/2+125, 240); ctx.arc_negative(W/2, 240, 125, 0, math.pi); ctx.close_path(); ctx.fill()
ctx.rectangle(W/2-65, 240, 130, 15); ctx.fill()                     # 눈썹(일자)

# ───────────────────────────── 눈
rgb((0,0,0)); ctx.set_line_width(3)
for dx in (-45, 45):                              # 안구 윤곽
    ctx.arc(W/2+dx, 260, 18, 0, 2*math.pi); ctx.stroke()
    ctx.arc(W/2+dx, 260, 7, 0, 2*math.pi); ctx.fill()  # 동공

# ───────────────────────────── 입술
rgb((0.39, 0.13, 0.15)); ctx.set_line_width(5)
ctx.move_to(W/2-35, 305); ctx.rel_curve_to(15, 15, 55, 15, 70, 0); ctx.stroke()

# ───────────────────────────── 손
hand = cairo.LinearGradient(0, 0, 0, 1)
hand.add_color_stop_rgb(0, *SKIN_LT); hand.add_color_stop_rgb(1, *SKIN_DK)
ctx.save(); ctx.translate(450, 700); ctx.rotate(math.radians(-10))
ctx.rectangle(0, 0, 240, 40); ctx.set_source(hand); ctx.fill()      # 팔
for i in range(5):
    ctx.rectangle(190+i*10, 30, 60, 15); ctx.fill()                # 손가락
ctx.restore()

# ───────────────────────────── 저장
surface.write_to_png("frida_pycairo.png")
print("done → frida_pycairo.png")
