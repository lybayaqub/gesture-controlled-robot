"""
Gesture-Controlled Robot — Dataset Simulation
===============================================
No webcam needed. Loads images from LeapGestRecog dataset,
runs them through your trained model, and drives the robot.


CONTROLS:
    SPACE       next image (manual mode)
    A           toggle auto / manual
    + / -       faster / slower auto cycle
    1–9, 0      jump to gesture class
    R           reset robot
    C           clear trail
    Q           quit
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os, sys, glob, random, time

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
MODEL_PATH   = "best_gesture_model.h5"
CLASSES_PATH = "gesture_classes.npy"
DATASET_ROOT = "leapGestRecog"          # change if dataset is elsewhere
IMG_SIZE     = 64
AUTO_DELAY   = 0.7                      # seconds per image in auto mode

GESTURE_KEYS = [
    "01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb",
    "06_index", "07_ok", "08_palm_moved", "09_c", "10_down",
]

GESTURE_COMMAND_MAP = {
    "01_palm":       "STOP",
    "02_l":          "TURN LEFT",
    "03_fist":       "MOVE FORWARD",
    "04_fist_moved": "MOVE BACKWARD",
    "05_thumb":      "SPEED UP",
    "06_index":      "MOVE FORWARD",
    "07_ok":         "MOVE BACKWARD",
    "08_palm_moved": "MOVE RIGHT",
    "09_c":          "TURN LEFT",
    "10_down":       "TURN RIGHT",
}

# BGR colours per command
CMD_COLOR = {
    "STOP":          (74,  75, 226),
    "TURN LEFT":     (221,119, 127),
    "MOVE FORWARD":  ( 29,158, 117),
    "MOVE BACKWARD": (221,138,  55),
    "SPEED UP":      ( 34,158,  29),
    "MOVE RIGHT":    ( 15,158, 110),
    "TURN RIGHT":    (126, 83, 212),
    "—":             (100,100, 100),
    "UNKNOWN":       (100,100, 100),
}

# Canvas sizes
LP_W, LP_H = 500, 560     # left panel  (image + prediction info)
RP_W, RP_H = 560, 560     # right panel (robot simulation)
GRID_N     = 10
MARGIN     = 50
CELL       = (RP_W - MARGIN * 2) // GRID_N
FONT       = cv2.FONT_HERSHEY_SIMPLEX


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def txt(img, text, pos, scale, color, thick=1):
    cv2.putText(img, str(text), pos, FONT, scale, color, thick, cv2.LINE_AA)

def hline(img, y, w, color=(40,40,40)):
    cv2.line(img, (0, y), (w, y), color, 1)

def filled_rect(img, x1, y1, x2, y2, color):
    cv2.rectangle(img, (x1,y1), (x2,y2), color, -1)

def border_rect(img, x1, y1, x2, y2, color, thick=1):
    cv2.rectangle(img, (x1,y1), (x2,y2), color, thick)

def conf_bar(img, x, y, w, h, value, color):
    filled_rect(img, x, y, x+w, y+h, (40,40,40))
    filled_rect(img, x, y, x+int(w*value), y+h, color)


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD MODEL & DATASET
# ══════════════════════════════════════════════════════════════════════════════
def load_assets():
    for p in (MODEL_PATH, CLASSES_PATH):
        if not os.path.exists(p):
            sys.exit(f"\n  ERROR: '{p}' not found in {os.getcwd()}\n"
                     "  Place model files next to this script.")
    print("Loading model ...", end=" ", flush=True)
    model   = load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    print(f"OK  |  {len(classes)} classes")
    return model, classes


def load_dataset():
    if not os.path.exists(DATASET_ROOT):
        sys.exit(
            f"\n  ERROR: Dataset folder '{DATASET_ROOT}' not found.\n"
            f"  Run this to download it:\n"
            f"    pip install kaggle\n"
            f"    kaggle datasets download gti-upm/leapgestrecog --unzip -p .\n"
        )
    print("Scanning dataset ...")
    dataset = {}
    for g in GESTURE_KEYS:
        paths = sorted(
            glob.glob(os.path.join(DATASET_ROOT, "**", g, "*.png"), recursive=True) +
            glob.glob(os.path.join(DATASET_ROOT, "**", g, "*.jpg"), recursive=True)
        )
        if paths:
            dataset[g] = paths
            print(f"  {g:<22} {len(paths):5d} images")
        else:
            print(f"  {g:<22}  NOT FOUND — check folder structure")
    if not dataset:
        sys.exit("  No images found. Check DATASET_ROOT path.")
    print(f"  Total: {sum(len(v) for v in dataset.values())} images\n")
    return dataset


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICT
# ══════════════════════════════════════════════════════════════════════════════
def predict_image(img_path, model, classes):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return "—", "—", 0.0, np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    inp     = resized.astype("float32") / 255.0
    inp     = inp.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    probs   = model.predict(inp, verbose=0)[0]
    idx     = int(np.argmax(probs))
    conf    = float(probs[idx])
    gesture = str(classes[idx])
    command = GESTURE_COMMAND_MAP.get(gesture, "UNKNOWN")
    return gesture, command, conf, resized


# ══════════════════════════════════════════════════════════════════════════════
#  ROBOT
# ══════════════════════════════════════════════════════════════════════════════
class Robot:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = self.y = 5.0
        self.heading  = 90
        self.step     = 0.4
        self.trail    = [(5.0, 5.0)]
        self.last_cmd = "IDLE"

    def clear_trail(self):
        self.trail = [(self.x, self.y)]

    def execute(self, command):
        self.last_cmd = command
        rad = np.radians(self.heading)
        if   command == "MOVE FORWARD":
            self.x += self.step * np.cos(rad)
            self.y += self.step * np.sin(rad)
        elif command == "MOVE BACKWARD":
            self.x -= self.step * np.cos(rad)
            self.y -= self.step * np.sin(rad)
        elif command == "TURN LEFT":
            self.heading = (self.heading + 15) % 360
        elif command == "TURN RIGHT":
            self.heading = (self.heading - 15) % 360
        elif command == "MOVE RIGHT":
            perp = np.radians(self.heading - 90)
            self.x += self.step * np.cos(perp)
            self.y += self.step * np.sin(perp)
        elif command == "SPEED UP":
            self.step = min(self.step + 0.1, 1.5)
        self.x = float(np.clip(self.x, 0, GRID_N))
        self.y = float(np.clip(self.y, 0, GRID_N))
        self.trail.append((self.x, self.y))
        if len(self.trail) > 800:
            self.trail.pop(0)


# ══════════════════════════════════════════════════════════════════════════════
#  LEFT PANEL — dataset image + prediction
# ══════════════════════════════════════════════════════════════════════════════
def draw_left_panel(raw_img, true_gest, pred_gest, command,
                    conf, img_path, auto_mode, auto_delay,
                    correct, total, all_stats):

    panel = np.full((LP_H, LP_W, 3), (14, 12, 12), dtype=np.uint8)
    col   = CMD_COLOR.get(command, (100,100,100))
    hit   = pred_gest == true_gest

    # ── header ────────────────────────────────────────────────────────────────
    filled_rect(panel, 0, 0, LP_W, 38, (22, 18, 18))
    txt(panel, "DATASET  IMAGE", (14, 14), 0.40, (70,70,70))
    txt(panel, true_gest.replace("_"," ").upper(), (14, 30), 0.52, (190,190,190), 1)

    # ── large image display ───────────────────────────────────────────────────
    if raw_img is not None and raw_img.size > 0:
        # scale up for display
        disp    = cv2.resize(raw_img, (240, 240), interpolation=cv2.INTER_NEAREST)
        disp3   = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

        # subtle command colour tint
        tint    = np.full_like(disp3, tuple(c//5 for c in col))
        disp3   = cv2.addWeighted(disp3, 0.90, tint, 0.10, 0)

        # border: green = correct, blue = wrong
        bc = (0,200,80) if hit else (80,80,210)
        border_rect(disp3, 0, 0, 239, 239, bc, 3)

        # paste centred horizontally
        px = (LP_W - 240) // 2
        panel[45:285, px:px+240] = disp3

        # 64×64 input thumbnail (top-right)
        thumb = cv2.resize(raw_img, (72,72), interpolation=cv2.INTER_NEAREST)
        thumb3 = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        border_rect(thumb3, 0, 0, 71, 71, (60,60,60))
        panel[45:117, LP_W-82:LP_W-10] = thumb3
        txt(panel, "model", (LP_W-80, 125), 0.28, (55,55,55))
        txt(panel, "input", (LP_W-80, 136), 0.28, (55,55,55))

    # ── prediction results ────────────────────────────────────────────────────
    ry = 295
    hline(panel, ry-8, LP_W)

    rows = [
        ("true  gesture", true_gest,             (170,170,170), False),
        ("pred  gesture", pred_gest,              (0,210,90) if hit else (80,80,220), True),
        ("robot command", command,                col,           True),
        ("confidence",    f"{conf:.1%}",          (150,150,150), False),
    ]
    for i, (label, value, vc, bold) in enumerate(rows):
        yy = ry + 8 + i*24
        txt(panel, label + " :", (14, yy), 0.42, (80,80,80))
        txt(panel, value,        (200, yy), 0.48, vc, 2 if bold else 1)

    # CORRECT / WRONG badge
    badge_col = (0,160,65) if hit else (65,65,195)
    badge_txt = "CORRECT" if hit else "WRONG"
    filled_rect(panel, LP_W-112, ry-6, LP_W-8, ry+18, badge_col)
    txt(panel, badge_txt, (LP_W-106, ry+10), 0.50, (255,255,255), 2)

    # ── confidence bar ────────────────────────────────────────────────────────
    conf_bar(panel, 14, ry+100, LP_W-28, 10, conf, col)
    txt(panel, f"{conf:.0%}", (LP_W-46, ry+108), 0.32, (120,120,120))

    # ── overall accuracy ──────────────────────────────────────────────────────
    acc     = correct/total*100 if total else 0
    acc_col = (0,200,80) if acc>=75 else (200,180,50) if acc>=50 else (80,80,200)
    txt(panel, f"accuracy: {acc:.0f}%  ({correct}/{total})",
        (14, ry+122), 0.42, acc_col)

    # ── per-gesture mini accuracy bars ───────────────────────────────────────
    bar_top = ry + 136
    bw      = (LP_W - 28) // len(GESTURE_KEYS)
    for i, g in enumerate(GESTURE_KEYS):
        c_ok, c_tot = all_stats.get(g, [0, 0])
        g_acc = c_ok/c_tot if c_tot else 0
        bh    = max(1, int(22 * g_acc))
        bx    = 14 + i * bw
        # background
        filled_rect(panel, bx, bar_top, bx+bw-2, bar_top+22, (30,30,30))
        # fill
        gc = (0, int(180*g_acc), int(80*g_acc))
        filled_rect(panel, bx, bar_top+22-bh, bx+bw-2, bar_top+22, gc)
        # key label
        key_lbl = str(i+1) if i < 9 else "0"
        txt(panel, key_lbl, (bx+3, bar_top+34), 0.28, (60,60,60))

    # current gesture highlight
    g_idx = GESTURE_KEYS.index(true_gest) if true_gest in GESTURE_KEYS else -1
    if g_idx >= 0:
        bx = 14 + g_idx * bw
        border_rect(panel, bx, bar_top, bx+bw-2, bar_top+22, (200,200,200), 1)

    # ── image path ────────────────────────────────────────────────────────────
    short = "…/" + "/".join(img_path.replace("\\","/").split("/")[-3:])
    txt(panel, short, (14, bar_top+50), 0.27, (40,40,40))

    # ── controls footer ───────────────────────────────────────────────────────
    hline(panel, LP_H-52, LP_W)
    mode_txt = f"AUTO  {auto_delay:.1f}s/img   +/- speed" if auto_mode \
               else "MANUAL   SPACE = next image"
    mode_col = (0,180,180) if auto_mode else (160,160,160)
    txt(panel, mode_txt,    (14, LP_H-34), 0.40, mode_col)
    txt(panel, "A toggle auto   1-9,0 gesture   R reset   C clear   Q quit",
        (14, LP_H-14), 0.28, (50,50,50))

    return panel


# ══════════════════════════════════════════════════════════════════════════════
#  RIGHT PANEL — robot simulation
# ══════════════════════════════════════════════════════════════════════════════
def draw_right_panel(robot, pred_gest, command, conf, cmd_log, correct, total):
    canvas = np.full((RP_H, RP_W, 3), (14,12,12), dtype=np.uint8)
    col    = CMD_COLOR.get(command, (100,100,100))

    # ── header ────────────────────────────────────────────────────────────────
    filled_rect(canvas, 0, 0, RP_W, 38, (22,18,18))
    txt(canvas, "ROBOT SIMULATION  —  top-down view", (14,26), 0.42, (60,60,60))

    # ── grid ──────────────────────────────────────────────────────────────────
    for i in range(GRID_N+1):
        gx = MARGIN + i*CELL
        gy = MARGIN + i*CELL
        cv2.line(canvas,(gx,MARGIN),(gx,MARGIN+GRID_N*CELL),(35,35,35),1)
        cv2.line(canvas,(MARGIN,gy),(MARGIN+GRID_N*CELL,gy),(35,35,35),1)

    # axis ticks
    for i in range(0, GRID_N+1, 2):
        txt(canvas, str(i),
            (MARGIN+i*CELL-4, MARGIN+GRID_N*CELL+16), 0.28, (50,50,50))
        txt(canvas, str(GRID_N-i),
            (MARGIN-22, MARGIN+i*CELL+4), 0.28, (50,50,50))

    # ── trail ─────────────────────────────────────────────────────────────────
    n = len(robot.trail)
    if n > 1:
        for i in range(1, n):
            a  = i / n
            tc = (int(20*a), int(90*a), int(220*a))
            p1 = (MARGIN+int(robot.trail[i-1][0]*CELL),
                  MARGIN+int((GRID_N-robot.trail[i-1][1])*CELL))
            p2 = (MARGIN+int(robot.trail[i][0]*CELL),
                  MARGIN+int((GRID_N-robot.trail[i][1])*CELL))
            cv2.line(canvas, p1, p2, tc, 2, cv2.LINE_AA)

    # ── start marker ─────────────────────────────────────────────────────────
    sx = MARGIN + int(5*CELL)
    sy = MARGIN + int(5*CELL)
    cv2.circle(canvas,(sx,sy),4,(50,50,50),-1)
    txt(canvas,"start",(sx+6,sy+4),0.26,(42,42,42))

    # ── robot ─────────────────────────────────────────────────────────────────
    rx = MARGIN + int(robot.x*CELL)
    ry = MARGIN + int((GRID_N-robot.y)*CELL)

    # outer glow
    cv2.circle(canvas,(rx,ry),22,tuple(max(0,c//6) for c in col),-1,cv2.LINE_AA)
    # body
    cv2.circle(canvas,(rx,ry),14,col,-1,cv2.LINE_AA)
    cv2.circle(canvas,(rx,ry),14,(255,255,255),1,cv2.LINE_AA)

    # direction arrow
    rad = np.radians(robot.heading)
    ax2 = int(rx + 28*np.cos(rad))
    ay2 = int(ry - 28*np.sin(rad))
    cv2.arrowedLine(canvas,(rx,ry),(ax2,ay2),(0,215,255),2,
                    tipLength=0.40, line_type=cv2.LINE_AA)

    # ── command badge ─────────────────────────────────────────────────────────
    bx,by = RP_W-210, 44
    filled_rect(canvas, bx, by, bx+198, by+38, col)
    txt(canvas, command, (bx+10,by+28), 0.72, (255,255,255), 2)
    txt(canvas, pred_gest.replace("_"," "), (bx,by+54), 0.36, (110,110,110))

    # ── confidence bar ────────────────────────────────────────────────────────
    conf_bar(canvas, MARGIN, 44, GRID_N*CELL, 10, conf, col)
    txt(canvas, f"conf {conf:.0%}", (MARGIN, 66), 0.30, (80,80,80))

    acc = correct/total*100 if total else 0
    acc_c = (0,190,70) if acc>=75 else (190,170,40) if acc>=50 else (80,80,190)
    txt(canvas, f"model acc {acc:.0f}%", (MARGIN+160,66), 0.30, acc_c)

    # ── HUD row ───────────────────────────────────────────────────────────────
    hud_y = MARGIN + GRID_N*CELL + 20
    hline(canvas, hud_y-8, RP_W)
    hud_items = [
        f"pos  ({robot.x:.1f}, {robot.y:.1f})",
        f"head  {robot.heading}°",
        f"spd  {robot.step:.1f}",
    ]
    for i, t in enumerate(hud_items):
        txt(canvas, t, (MARGIN + i*170, hud_y+8), 0.36, (120,120,120))

    # ── command log ───────────────────────────────────────────────────────────
    log_y = hud_y + 28
    txt(canvas, "log", (MARGIN, log_y), 0.30, (45,45,45))
    for i, entry in enumerate(reversed(cmd_log[-7:])):
        b = 50 + i*25
        txt(canvas, entry, (MARGIN, log_y+14+i*15), 0.30, (b,b,b))

    return canvas


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    model, classes = load_assets()
    dataset        = load_dataset()
    robot          = Robot()

    cmd_log    = []
    auto_mode  = True
    auto_delay = AUTO_DELAY
    correct    = 0
    total      = 0
    all_stats  = {g: [0, 0] for g in GESTURE_KEYS}

    # start state
    cur_gesture  = GESTURE_KEYS[0]
    img_list     = random.sample(dataset[cur_gesture], len(dataset[cur_gesture]))
    img_idx      = 0
    pred_gesture = "—"
    command      = "—"
    conf         = 0.0
    raw_img      = np.zeros((IMG_SIZE,IMG_SIZE), dtype=np.uint8)
    img_path     = ""

    def switch_gesture(g):
        nonlocal cur_gesture, img_list, img_idx
        cur_gesture = g
        img_list    = random.sample(dataset[g], len(dataset[g]))
        img_idx     = 0
        print(f"  → {g}")

    def load_next():
        nonlocal img_idx, pred_gesture, command, conf, raw_img, img_path
        nonlocal correct, total
        if img_idx >= len(img_list):
            img_idx = 0
            random.shuffle(img_list)
        img_path = img_list[img_idx];  img_idx += 1
        pred_gesture, command, conf, raw_img = predict_image(img_path, model, classes)
        total += 1
        all_stats[cur_gesture][1] += 1
        if pred_gesture == cur_gesture:
            correct += 1
            all_stats[cur_gesture][0] += 1
        robot.execute(command)
        ts    = time.strftime("%H:%M:%S")
        entry = f"{ts}  {command:<15} {pred_gesture}"
        cmd_log.append(entry)
        if len(cmd_log) > 80: cmd_log.pop(0)
        hit = "✓" if pred_gesture == cur_gesture else "✗"
        print(f"  {hit} [{cur_gesture}] → pred={pred_gesture:<22} "
              f"cmd={command:<15} conf={conf:.0%}")

    load_next()
    last_auto = time.time()

    print("="*60)
    print("  Running  |  two windows open")
    print("  SPACE next   A auto   +/- speed   1-9,0 gesture")
    print("  R reset   C clear   Q quit")
    print("="*60)

    while True:
        now = time.time()

        if auto_mode and (now - last_auto) >= auto_delay:
            last_auto = now
            load_next()

        left  = draw_left_panel(
            raw_img, cur_gesture, pred_gesture, command, conf,
            img_path, auto_mode, auto_delay, correct, total, all_stats
        )
        right = draw_right_panel(
            robot, pred_gesture, command, conf, cmd_log, correct, total
        )


        # stitch side by side
        combined = np.hstack([left, right])
        cv2.imshow("Gesture Robot  —  Dataset Simulation", combined)

        key = cv2.waitKey(30) & 0xFF

        if   key == ord('q'): break

        elif key == ord(' '):
            load_next()
            last_auto = time.time()

        elif key == ord('a'):
            auto_mode = not auto_mode
            print(f"  Auto: {'ON' if auto_mode else 'OFF'}")

        elif key in (ord('+'), ord('=')):
            auto_delay = max(0.1, round(auto_delay-0.1, 1))
            print(f"  Speed: {auto_delay}s/img")

        elif key == ord('-'):
            auto_delay = min(3.0, round(auto_delay+0.1, 1))
            print(f"  Speed: {auto_delay}s/img")

        elif key == ord('r'):
            robot.reset()
            print("  Robot reset.")

        elif key == ord('c'):
            robot.clear_trail()
            print("  Trail cleared.")

        else:
            key_map = {ord(str(i+1)): GESTURE_KEYS[i] for i in range(9)}
            if len(GESTURE_KEYS) >= 10:
                key_map[ord('0')] = GESTURE_KEYS[9]
            if key in key_map:
                switch_gesture(key_map[key])
                load_next()
                last_auto = time.time()

    cv2.destroyAllWindows()

    # ── final report ──────────────────────────────────────────────────────────
    acc = correct/total*100 if total else 0
    print(f"\n{'='*45}")
    print(f"  Final accuracy : {acc:.1f}%  ({correct}/{total})")
    print(f"{'='*45}")
    print("  Per-gesture breakdown:")
    for g in GESTURE_KEYS:
        c_ok, c_tot = all_stats[g]
        g_acc = c_ok/c_tot*100 if c_tot else 0
        bar   = "█" * int(g_acc/5) + "░" * (20-int(g_acc/5))
        print(f"    {g:<22} {bar}  {g_acc:5.1f}%  ({c_ok}/{c_tot})")
    print()


if __name__ == "__main__":
    main()