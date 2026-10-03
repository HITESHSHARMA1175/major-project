"""
show_my_signs.py
================
Loads hello.png, thanku.png, loveyou.png and displays them
exactly like the notebook's prob_viz cell — but with YOUR images.
Also saves a combined image: my_signs.png
"""

import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')          # use a GUI backend
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ACTIONS = np.array(['hello', 'thanks', 'iloveyou'])
COLORS  = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

# ── Map each image file to its action index ───────────────────────────────────
IMAGE_MAP = [
    (os.path.join(BASE_DIR, 'hello.png'),   0),   # hello   → index 0
    (os.path.join(BASE_DIR, 'thanku.png'),  1),   # thanks  → index 1
    (os.path.join(BASE_DIR, 'loveyou.png'), 2),   # iloveyou→ index 2
]


def prob_viz(res, actions, input_frame, colors):
    """Draw probability bars on the left side of the frame."""
    out = input_frame.copy()
    for i, prob in enumerate(res):
        cv2.rectangle(out, (0, 60 + i*40), (int(prob * 100), 90 + i*40), colors[i], -1)
        cv2.putText(out, actions[i], (0, 85 + i*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def add_banner(image, label):
    """Add top banner with detected label."""
    out = image.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 45), (245, 117, 16), -1)
    cv2.putText(out, f'Detected: {label.upper()}', (8, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    fig.patch.set_facecolor('#1e1e1e')
    fig.suptitle('Sign Language Detection — My Signs',
                 fontsize=20, color='white', fontweight='bold', y=1.01)

    for ax, (img_path, action_idx) in zip(axes, IMAGE_MAP):
        if not os.path.exists(img_path):
            print(f"[!] File not found: {img_path}")
            ax.axis('off')
            continue

        # Load image (BGR) and resize to standard size
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"[!] Could not read: {img_path}")
            ax.axis('off')
            continue
        bgr = cv2.resize(bgr, (640, 480))

        # Build a fake probability vector: 100% for the correct action
        res = np.zeros(len(ACTIONS))
        res[action_idx] = 1.0

        # Overlay prob bars + banner
        out = prob_viz(res, ACTIONS, bgr, COLORS)
        out = add_banner(out, ACTIONS[action_idx])

        # Convert BGR → RGB for matplotlib
        rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        ax.imshow(rgb)
        ax.set_title(ACTIONS[action_idx].upper(),
                     fontsize=16, color='white', fontweight='bold', pad=10)
        ax.axis('off')

    plt.tight_layout()

    # Save combined image
    out_path = os.path.join(BASE_DIR, 'my_signs.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"[OK] Saved combined image: {out_path}")

    plt.show()
    print("[OK] Done.")


if __name__ == '__main__':
    main()
