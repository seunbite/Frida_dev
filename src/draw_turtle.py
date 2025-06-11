# frida_turtle.py ── stylised Python-Turtle redraw
import turtle

# ────────── 색 팔레트(대략) ──────────
DEEP_NAVY   = "#0b1a2a"   # 배경 밤하늘/물결
TEAL_LINE   = "#0f4458"   # 물결 윤곽선
SKIN        = "#e9b4a3"
CHEEK       = "#d6847e"
LIP         = "#802b2e"
HAIR        = "#1b1b1b"
ROBES       = "#6a1024"
GOLD        = "#d8a73b"
SCARLET     = "#c11c1d"
BLACK       = "#000000"

screen = turtle.Screen()
screen.setup(900, 1200)
screen.bgcolor(DEEP_NAVY)
screen.title("Stylised Turtle portrait")

t = turtle.Turtle(visible=False)
t.speed(0)
t.pensize(3)

# ────────── 유틸 함수 ──────────
def move(x, y):
    t.penup(); t.goto(x, y); t.pendown()

def filled(draw_fn, color):
    t.color(color); t.begin_fill(); draw_fn(); t.end_fill()

# ────────── 배경 물결 ──────────
def draw_waves():
    move(-450, -50)
    t.color(TEAL_LINE); t.pensize(4)
    for _ in range(2):
        for _ in range(5):
            t.circle(60, 90)      # 위로 소용돌이
            t.circle(-60, 90)     # 아래로 소용돌이
        t.forward(900); t.right(180)
    t.pensize(3)

# ────────── 로브(상의) ──────────
def draw_robe():
    def body():
        move(-200, -400)
        for heading, dist in [(-90, 450), (0, 400), (90, 450), (180, 400)]:
            t.setheading(heading); t.forward(dist)
    filled(body, ROBES)

    # 금색 트리밍
    for side in (-1, 1):
        move(side*60, 170)
        t.pensize(18); t.color(GOLD)
        t.setheading(-90)
        t.circle(60*side, 180)
    t.pensize(3)

    # 안쪽 무늬 (검정/적)
    for side in (-1, 1):
        move(side*45, 170); t.setheading(-90)
        pattern = [("arc", 50*side, 45), ("line", 70),
                   ("arc", 30*side, 45), ("line", 70)]
        for cmd, val, *rest in pattern*2:
            if cmd == "arc":
                t.color(SCARLET); t.circle(val, 90)
            else:
                t.color(BLACK); t.forward(val)

# ────────── 목 / 얼굴 ──────────
def draw_neck_and_face():
    filled(lambda: (
        move(-40, 100), t.setheading(-90), t.forward(140),
        t.setheading(0), t.forward(80),
        t.setheading(90), t.forward(140)
    ), SKIN)

    filled(lambda: (
        move(0, 240), t.setheading(0), t.circle(100)
    ), SKIN)

# ────────── 머리카락 / 눈썹 ──────────
def draw_hair_and_brows():
    filled(lambda: (
        move(0, 340), t.setheading(0), t.circle(100, 180),
        t.setheading(-90), t.forward(50),
        t.circle(-50, 180), t.forward(50)
    ), HAIR)

    # 가르마
    move(0, 340); t.color(SKIN); t.setheading(-90); t.pensize(6); t.forward(60)
    # 눈썹
    move(-60, 240); t.color(HAIR); t.setheading(0); t.pensize(10); t.forward(120)
    t.pensize(3)

# ────────── 눈 / 입술 / 볼 터치 ──────────
def draw_features():
    for dx in (-40, 40):
        move(dx, 210); t.dot(25, SKIN); t.dot(10, BLACK)  # 눈
    move(0, 170); t.color(LIP); t.setheading(-10); t.pensize(4); t.circle(35, 40)
    t.pensize(3)
    for dx in (-55, 55):
        move(dx, 190); t.dot(60, CHEEK)  # 볼

# ────────── 손 ──────────
def draw_hand():
    filled(lambda: (
        move(130, -50), t.setheading(-10), t.forward(280),
        t.setheading(70), t.forward(40),
        t.setheading(190), t.forward(280),
        t.setheading(-110), t.forward(40)
    ), SKIN)

# ────────── 메인 호출 ──────────
for fn in (draw_waves, draw_robe,
           draw_neck_and_face, draw_hair_and_brows,
           draw_features, draw_hand):
    fn()

turtle.done()
