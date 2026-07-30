"""
pipeline_worker.py

Runs the actual frame-extraction / classification pipeline as a standalone
script, invoked via `subprocess.run([sys.executable, "pipeline_worker.py", ...])`
from the notebook.
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
from fastai.vision.all import load_learner, defaults


@contextmanager
def set_windows_posix():
    windows_backup = pathlib.WindowsPath
    try:
        pathlib.WindowsPath = pathlib.PosixPath
        yield
    finally:
        pathlib.WindowsPath = windows_backup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coords", required=True, help="Path to coordinates CSV")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--trained-model", required=True, help="Path to fastai .pkl model")
    parser.add_argument("--weights", required=True, help="Path to YOLOv5 weights .pt file")
    parser.add_argument("--time-interval", type=int, default=5, help="Frame extraction interval, seconds")
    args = parser.parse_args()

    print("🚀 System: Libraries imported successfully.")

    # --- 1. SET UP OUTPUT FOLDERS ---
    folder_out = "./output/video_frames_test"
    if os.path.exists(folder_out):
        shutil.rmtree(folder_out)
    os.makedirs(folder_out)

    folder_out_yolo = "./output/yolo_predict"
    if os.path.exists(folder_out_yolo):
        shutil.rmtree(folder_out_yolo)
    os.makedirs(folder_out_yolo)

    # --- 2. LOAD & PARSE DATAFRAME DATA ---
    coords = pd.read_csv(args.coords)
    coords["datetime"] = coords["date"] + " " + coords["time"]
    coords["datetime"] = coords["datetime"].astype("datetime64[ns]")
    coords["UNIXtime"] = (coords["datetime"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1millisecond")
    coords = coords.drop(columns=["date", "time"])
    coords = coords.set_index("datetime", drop=True)

    creation_time_datetime = coords.index[0]

    # --- 3. PRE-COMPUTE COORDINATES IN MEMORY FOR SPEED ---
    print("Caching coordinate data for fast interpolation...")
    coord_unix = coords["UNIXtime"].to_numpy()
    coord_lat = coords["latitude"].to_numpy()
    coord_lon = coords["longitude"].to_numpy()

    # --- 4. DETECT & SET GPU DEVICE ---
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        defaults.device = device  
        yolo_device = "0"  
        print(f"🚀 Inference Device: {torch.cuda.get_device_name(0)} (GPU)")
    else:
        device = torch.device("cpu")
        defaults.device = torch.device("cpu")
        yolo_device = "cpu"
        print("⚠️ WARNING: GPU not found. Running on CPU will be extremely slow!")

    # --- 5. LOAD MODELS ONTO GPU ---
    with set_windows_posix():
    learner = load_learner(args.trained_model, cpu=(device.type == "cpu"))

    
    learner.dls.device = device
    learner.model = learner.model.to(device) 
    
    model_yolo = torch.hub.load("yolov5", "custom", path=args.weights, source="local", device=yolo_device)
    model_yolo.conf = 0.7

    # --- 6. INITIALISE RESULTS ARRAYS ---
    timestamp_list = []
    filename_list = []
    prediction_list = []
    latitude_list = []
    longitude_list = []
    image_path_list = []
    burrowing_sea_cucumber_list = []
    horse_mussel_list = []
    northern_sea_fan_list = []

    # --- 7. VIDEO STREAM CONFIGURATION ---
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: 
        fps = 30.0  
    
    frame_step = int(fps * args.time_interval)
    frame_count = 0
    
    print("Processing video stream via GPU pipeline...")
    
    # Disable gradient tracking across the entire loop
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_step == 0:
                # High-speed timestamp math
                frame_seconds = frame_count / fps
                new_frame_timestamp_datetime = creation_time_datetime + timedelta(seconds=frame_seconds)
                new_frame_timestamp_unix = int(
                    (new_frame_timestamp_datetime - pd.Timestamp("1970-01-01")).total_seconds() * 1000
                )
                new_frame_timestamp = new_frame_timestamp_datetime.strftime("%Y-%m-%d_%H-%M-%S.%f")

                # Fast image channel conversion
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)

                # GPU Model Inferencing
                pred, _, _ = learner.predict(frame_pil)
                pred_label = "Yes" if pred == "maerl" else "No"
                
                yolo_results = model_yolo(frame_pil)

                # Extract class counts directly from CUDA output arrays
                if len(yolo_results.xyxy[0]) > 0:
                    detected_classes = yolo_results.xyxy[0][:, 5].cpu().numpy().astype(int)
                else:
                    detected_classes = []
                
                burrowing_sea_cucumber = "Yes" if 0 in detected_classes else "No"
                horse_mussel = "Yes" if 1 in detected_classes else "No"
                northern_sea_fan = "Yes" if 2 in detected_classes else "No"

                # Fast memory-cached coordinate interpolation
                latitude = np.interp(new_frame_timestamp_unix, coord_unix, coord_lat)
                longitude = np.interp(new_frame_timestamp_unix, coord_unix, coord_lon)

                filename = f"Frame_{new_frame_timestamp}_{frame_count:04d}.jpg"
                image_path = os.path.join(folder_out, filename)
                
                timestamp_list.append(new_frame_timestamp)
                filename_list.append(filename)
                prediction_list.append(pred_label)
                image_path_list.append(image_path)
                burrowing_sea_cucumber_list.append(burrowing_sea_cucumber)
                horse_mussel_list.append(horse_mussel)
                northern_sea_fan_list.append(northern_sea_fan)
                latitude_list.append(latitude)
                longitude_list.append(longitude)

                # 🛑 OPTIONAL DISK SAVING: Uncomment to visually review frame detections on disk.
                # WARNING: Will slow performance depending on drive speeds.
                frame_pil.save(image_path)
                image_path_yolo = os.path.join(folder_out_yolo, filename)
                Image.fromarray(yolo_results.render()[0]).save(image_path_yolo)

                # Hard fast-forward the video capture buffer to skip processing frames we don't look at
                frame_count += frame_step
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                continue

            frame_count += 1

    cap.release()

    # --- 8. BUILD AND EXPORT LOG CSV ---
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

    csv_save_name = Path(args.video).stem + "_out.csv"
    output_df.to_csv(csv_save_name, index=False)

    print(f"\n{'=' * 50}")
    print("✅ Processing complete")
    print(f"Frames processed: {len(filename_list)}")
    print(f"Output CSV: {csv_save_name}")
    print(f"Detections — BurrowingSeaCucumber: {burrowing_sea_cucumber_list.count('Yes')}, "
          f"HorseMussel: {horse_mussel_list.count('Yes')}, "
          f"NorthernSeaFan: {northern_sea_fan_list.count('Yes')}, "
          f"Maerl: {prediction_list.count('Yes')}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
