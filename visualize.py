#!/usr/bin/env python3
"""
Colab‑friendly viewer for all .jpg / .png files produced by YOLOv8
(trains, preds, PR curves, confusion matrices…).

Usage
-----
!python visualize.py                     # auto‑detect runs folder
!python visualize.py /content/runs       # explicit path
!python visualize.py runs/detect/train   # show only training previews
"""

import os
import sys
import glob
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import ipywidgets as widgets
from PIL import Image
from IPython.display import display


# --------------------------------------------------
# 1.  CONFIG & PATH DETECTION
# --------------------------------------------------
SUPPORTED_EXT = [".jpg", ".png"]


def guess_runs_dirs() -> List[Path]:
    """Return a list of plausible runs/ locations (ordered by likelihood)."""
    here = Path.cwd()

    candidates = [
        here / "runs",                               # ./runs
        here.parent / "runs",                        # ../runs
        Path("/content/runs"),                       # Colab default when cwd=/content
        Path("/content/drive/MyDrive") / "runs",     # Some users cd into random sub‑folders
    ]

    # Also look two levels up for runs/**/detect
    for p in list(here.parents)[:3]:
        candidates.append(p / "runs")

    # uniq & keep order
    seen, uniq = set(), []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def first_dir_with_images(paths: List[Path]) -> Path:
    """Return first directory that actually contains images (recursively)."""
    for path in paths:
        if path.exists():
            imgs = []
            for ext in SUPPORTED_EXT:
                imgs.extend(path.rglob(f"*{ext}"))
            if imgs:
                return path
    return None


# --------------------------------------------------
# 2.  VIEWER CLASS
# --------------------------------------------------
class RunsVisualizer:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.image_files = self._find_images()
        self.index = 0
        if not self.image_files:
            raise FileNotFoundError(
                f"No .jpg/.png files under {base_dir}.\n"
                f"(looked for {SUPPORTED_EXT})"
            )
        print(f"✅ Found {len(self.image_files)} images under {base_dir}")

    # ----------------------------------------------
    def _find_images(self):
        files = []
        for ext in SUPPORTED_EXT:
            files.extend(self.base_dir.rglob(f"*{ext}"))
        return sorted(files)

    # ----------------------------------------------
    def _show(self):
        img_path = self.image_files[self.index]
        img = Image.open(img_path)

        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis("off")
        rel = img_path.relative_to(self.base_dir)
        plt.title(f"[{self.index+1}/{len(self.image_files)}]  {rel}")
        plt.show()

    # ----------------------------------------------
    def run(self):
        next_btn = widgets.Button(description="Next ▶️")
        prev_btn = widgets.Button(description="◀️ Previous")
        out = widgets.Output()

        def refresh(_=None):
            out.clear_output(wait=True)
            with out:
                self._show()

        def _next(_):
            self.index = (self.index + 1) % len(self.image_files)
            refresh()

        def _prev(_):
            self.index = (self.index - 1) % len(self.image_files)
            refresh()

        next_btn.on_click(_next)
        prev_btn.on_click(_prev)

        display(widgets.HBox([prev_btn, next_btn]), out)
        refresh()


# --------------------------------------------------
# 3.  MAIN ENTRY POINT
# --------------------------------------------------
def main():
    # If user supplied a path → use it.  Otherwise guess.
    if len(sys.argv) > 1:
        dir_path = Path(sys.argv[1]).expanduser().resolve()
        search_paths = [dir_path]
    else:
        search_paths = guess_runs_dirs()

    chosen = first_dir_with_images(search_paths)
    if chosen is None:
        print("❌ No images found. Tried these directories:")
        for p in search_paths:
            print("   •", p)
        print("\nMake sure you trained first and/or pass the correct path:\n"
              "   !python visualize.py /content/your/folder")
        sys.exit(1)

    print("🔍 Visualizing images from:", chosen)
    RunsVisualizer(chosen).run()


if __name__ == "__main__":
    main()
