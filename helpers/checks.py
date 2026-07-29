import os


def check_required_files(input_dir='./input',
                          accepted_coord_formats=('.csv', '.txt'),
                          accepted_video_formats=('.mp4', '.mov', '.m2ts', '.avi'),
                          prompt=True):
    """
    Verifies a coordinate file and a video file are present in input_dir.
    Returns (coords_path, video_path) on success, raises FileNotFoundError otherwise.
    """
    if prompt:
        input(f"Please upload your coordinate file {accepted_coord_formats} "
              f"and video file {accepted_video_formats} to the {input_dir} directory, "
              f"then press Enter to continue...")

    got_coord_file = False
    got_video_file = False
    coords = None
    inputVideo = None

    if not os.path.exists(input_dir):
        raise FileNotFoundError(
            f"Directory {input_dir} does not exist. Please create it and upload your files."
        )

    files = os.listdir(input_dir)
    print(f"Files in directory: {files}")

    for file in files:
        if file.endswith(tuple(accepted_coord_formats)):
            coords = os.path.join(input_dir, file)
            got_coord_file = True
        if file.endswith(tuple(accepted_video_formats)):
            inputVideo = os.path.join(input_dir, file)
            got_video_file = True

    if not got_coord_file or not got_video_file:
        raise FileNotFoundError(
            f"STOP! You need to upload both a coordinate file ({', '.join(accepted_coord_formats)}) "
            f"and a video file ({', '.join(accepted_video_formats)}) to the {input_dir} directory before proceeding."
        )

    print(f"All required files are present. Coordinate file: {coords}, Video file: {inputVideo}. You may proceed.")
    return coords, inputVideo