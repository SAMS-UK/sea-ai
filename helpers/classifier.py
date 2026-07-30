"""
pipeline_worker.py

Runs the actual frame-extraction / classification pipeline as a standalone
script, invoked via `subprocess.run([sys.executable, "pipeline_worker.py", ...])`
from the notebook.

Why this exists as a separate process rather than running inline in the
notebook's kernel: Colab's kernel has numpy already imported (as part of its
own startup) before any of the notebook's own pip installs run. Once a
module with a compiled C extension like numpy is imported into a running
process, reinstalling it on disk does nothing — the already-loaded module
stays cached in memory for the life of that process. A subprocess is a
genuinely new OS process with an empty module cache, so it imports the
pinned numpy==1.26.4 (and pandas/fastai/etc.) fresh and correctly, with no
kernel restart required.
"""
import argparse
import os
import shutil
import pathlib
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from fastai.vision.all import load_learner


@contextmanager
def set_posix_windows():
    """Needed if the model was exported on Linux and is being loaded elsewhere."""
    posix_backup = pathlib.PosixPath
    try:
        pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coords", required=True, help="Path to coordinates CSV")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--trained-model", required=True, help="Path to fastai .pkl model")
    parser.add_argument("--weights", required=True, help="Path to YOLOv5 weights .pt file")
    parser.add_argument("--time-interval", type=int, default=5, help="Frame extraction interval, seconds")
    args = parser.parse_args()

    print("imported libraries")

    coords = pd.read_csv(args.coords)
    coords["datetime"] = coords["date"] + " " + coords["time"]
    coords["datetime"] = coords["datetime"].astype("datetime64[ns]")
    coords["UNIXtime"] = (coords["datetime"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1millisecond")
    coords = coords.drop(columns=["date", "time"])
    coords = coords.set_index("datetime", drop=True)

    folder_out = "./output/video_frames_test"
    if os.path.exists(folder_out):
        shutil.rmtree(folder_out)
    os.makedirs(folder_out)

    folder_out_yolo = "./output/yolo_predict"
    if os.path.exists(folder_out_yolo):
        shutil.rmtree(folder_out_yolo)
    os.makedirs(folder_out_yolo)

    cap = cv2.VideoCapture(args.video)

    creation_time_datetime = coords.index[0]

    timestamp_list = []
    filename_list = []
    prediction_list = []
    latitude_list = []
    longitude_list = []
    image_path_list = []
    burrowing_sea_cucumber_list = []
    horse_mussel_list = []
    northern_sea_fan_list = []


    # --- DETECT & SET GPU DEVICE ---
    import torch
    
    if torch.cuda.is_available():
        #device = torch.device("cuda:0")
        device = torch.device(0)
        yolo_device = "0"  # YOLOv5 prefers the explicit string index "0" over "cuda"
        print(f"🚀 Inference Device: {torch.cuda.get_device_name(0)} (GPU)")
    else:
        device = torch.device("cpu")
        yolo_device = "cpu"
        print("⚠️ WARNING: GPU not found. Running on CPU will be extremely slow!")

    # --- LOAD MODELS ONTO GPU ---
    learner = load_learner(args.trained_model)
    learner.model.to(device) 
    
    # Pass yolo_device ("0" or "cpu") instead of device.type
    model_yolo = torch.hub.load("yolov5", "custom", path=args.weights, source="local", device=yolo_device)
    # ------------------------------

    


    
    frame_index = 0
    current_time = 0
    print("processing video!")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if frame_timestamp >= current_time:
            new_frame_timestamp_datetime = creation_time_datetime + timedelta(seconds=frame_timestamp)
            new_frame_timestamp_unix = (
                new_frame_timestamp_datetime - pd.Timestamp("1970-01-01")
            ) // pd.Timedelta("1millisecond")
            new_frame_timestamp = new_frame_timestamp_datetime.strftime("%Y-%m-%d_%H-%M-%S.%f")

            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


            pred, _, _ = learner.predict(frame_pil)
            pred = "Yes" if pred == "maerl" else "No"
            
            model_yolo.conf = 0.7
            yolo_results = model_yolo(frame_pil)

            filename = f"Frame_{new_frame_timestamp}_{frame_index:04d}.jpg"
            image_path = os.path.join(folder_out, filename)
            frame_pil.save(image_path)

            image_path_yolo = os.path.join(folder_out_yolo, filename)
            Image.fromarray(yolo_results.render()[0]).save(image_path_yolo)

            if not yolo_results.pandas().xyxy[0].empty:
                classes = np.array(yolo_results.pandas().xyxy[0]["class"])
                burrowing_sea_cucumber = "Yes" if 0 in classes else "No"
                horse_mussel = "Yes" if 1 in classes else "No"
                northern_sea_fan = "Yes" if 2 in classes else "No"
            else:
                burrowing_sea_cucumber = "No"
                horse_mussel = "No"
                northern_sea_fan = "No"

            timestamp_list.append(new_frame_timestamp)
            filename_list.append(filename)
            prediction_list.append(pred)
            image_path_list.append(image_path)
            burrowing_sea_cucumber_list.append(burrowing_sea_cucumber)
            horse_mussel_list.append(horse_mussel)
            northern_sea_fan_list.append(northern_sea_fan)

            matching_row = coords.loc[coords.index == new_frame_timestamp_datetime]
            if not matching_row.empty:
                latitude = matching_row["latitude"].iloc[0]
                longitude = matching_row["longitude"].iloc[0]
            else:
                latitude = np.interp(new_frame_timestamp_unix, coords["UNIXtime"], coords["latitude"])
                longitude = np.interp(new_frame_timestamp_unix, coords["UNIXtime"], coords["longitude"])

            latitude_list.append(latitude)
            longitude_list.append(longitude)

            frame_index += 1
            current_time += args.time_interval

    output_df = pd.DataFrame(
        {
            "Time": timestamp_list,
            "Filename": filename_list,
            "Latitude": latitude_list,
            "Longitude": longitude_list,
            "Path": image_path_list,
            "BurrowingSeaCucumber": burrowing_sea_cucumber_list,
            "HorseMussel": horse_mussel_list,
            "NorthernSeaFan": northern_sea_fan_list,
            "Maerl": prediction_list,
        }
    )

    cap.release()

    csv_save_name = Path(args.video).stem + "_out.csv"
    output_df.to_csv(csv_save_name)

    print(f"\n{'=' * 50}")
    print("✅ Processing complete")
    print(f"Frames processed: {len(filename_list)}")
    print(f"Output CSV: {csv_save_name}")
    print(f"Frames saved to: {folder_out}")
    print(f"YOLO-annotated frames saved to: {folder_out_yolo}")
    print(
        f"Detections — BurrowingSeaCucumber: {burrowing_sea_cucumber_list.count('Yes')}, "
        f"HorseMussel: {horse_mussel_list.count('Yes')}, "
        f"NorthernSeaFan: {northern_sea_fan_list.count('Yes')}, "
        f"Maerl: {prediction_list.count('Yes')}"
    )
    print(f"{'=' * 50}\n")
    print(output_df.to_string())


if __name__ == "__main__":
    main()
