import subprocess
import sys


def run_pipeline(coords, video, trained_model, weights, time_interval,
                  worker_script="sea-ai/helpers/model_run.py"):
    """
    Runs pipeline_worker.py in an isolated subprocess. See that file's own
    docstring for why: Colab's kernel pre-imports numpy at startup, so pip
    installing a pinned version afterward has no effect on the running kernel
    — only a fresh process picks it up.

    Streams output live (rather than capturing and printing at the end) so
    progress is visible during long-running video processing.
    """
    print("🚀 Running pipeline in an isolated subprocess...")

    process = subprocess.Popen(
        [sys.executable, worker_script,
         "--coords", coords,
         "--video", video,
         "--trained-model", trained_model,
         "--weights", weights,
         "--time-interval", str(time_interval)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end="")

    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError("Pipeline subprocess failed — see output above")

    print("\n✅ Pipeline subprocess complete")