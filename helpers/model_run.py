import os
import subprocess
import sys
import torch


def run_pipeline(
    coords,
    video,
    trained_model,
    weights,
    time_interval,
    worker_script="sea-ai/helpers/classifier.py",
):
    """Runs pipeline_worker.py in an isolated subprocess.

    Streams output live so progress is visible during long-running video
    processing.
    """
    print("=== PARENT ENVIRONMENT CUDA INFO ===")
    print(f"Python Executable: {sys.executable}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current Device ID: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(
        f"CUDA_VISIBLE_DEVICES Env Var: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}"
    )
    print("====================================\n")

    print("🚀 Running pipeline in an isolated subprocess...")

    # Explicitly copy environment variables so the subprocess inherits GPU visibility
    env = os.environ.copy()

    process = subprocess.Popen(
        [
            sys.executable,
            worker_script,
            "--coords",
            coords,
            "--video",
            video,
            "--trained-model",
            trained_model,
            "--weights",
            weights,
            "--time-interval",
            str(time_interval),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,  # Added env mapping to pass CUDA context down
    )

    for line in process.stdout:
        print(line, end="")

    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError("Pipeline subprocess failed — see output above")

    print("\n✅ Pipeline subprocess complete")
