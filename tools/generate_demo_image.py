from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    out = Path("data/input/demo.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    img = np.full((480, 640, 3), 255, dtype=np.uint8)
    # One-pixel strokes reduce inner/outer dual-boundary ambiguity in contour extraction.
    cv2.circle(img, (150, 150), 80, (0, 0, 0), 1)
    cv2.line(img, (40, 300), (280, 380), (0, 0, 0), 1)
    cv2.ellipse(img, (420, 160), (120, 70), 25, 0, 360, (0, 0, 0), 1)
    cv2.ellipse(img, (420, 340), (100, 100), 0, 30, 230, (0, 0, 0), 1)

    cv2.imwrite(str(out), img)
    print(str(out.resolve()))


if __name__ == "__main__":
    main()
