# setup.py
import contextlib
import io
import os
import shutil
import subprocess
import sys


def install_deps():
    print("⏳ Installing pinned dependencies and cloning YOLOv5...", flush=True)

    # 1. Pinned dependencies
    dependencies = [
        "numpy==1.26.4", "pandas==2.2.2", "scipy<1.14",
        "fastai<2.8.0", "fastcore<1.8.0", "opencv-python-headless",
        "ffmpeg-python", "Pillow", "wget", "requests",
        "mplleaflet", "seaborn", "matplotlib", "ultralytics"
    ]

    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        # Install packages silently
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-cache-dir"] + dependencies,
            capture_output=True, text=True
        )

    if result.returncode != 0:
        print("\n❌ Dependency install failed!", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    # 2. Idempotent git clone (Run from /content context)
    if os.path.exists("yolov5"):
        shutil.rmtree("yolov5")

    # Run git clone silently
    clone_result = subprocess.run(
        ["git", "clone", "https://github.com/ultralytics/yolov5", "yolov5"],
        capture_output=True, text=True
    )

    if clone_result.returncode != 0:
        print("\n❌ Repository clone failed!", file=sys.stderr)
        print(clone_result.stderr, file=sys.stderr)
        sys.exit(1)

    print("✅ Environment setup complete. YOLOv5 cloned and ready.", flush=True)


if __name__ == "__main__":
    install_deps()
