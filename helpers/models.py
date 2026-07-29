import os
import wget


WEIGHTS_URL = "https://thredds.sams.ac.uk/thredds/fileServer/models/weights/best_FE_2023-08-07_22-36.pt"
TRAINED_MODEL_URL = "https://thredds.sams.ac.uk/thredds/fileServer/models/trained/F-JuraIslay_model_binary_Mixup_b7_bs4_720_1280_01_08.pkl"

TRAINED_EXTENSIONS = (".pkl", ".h5", ".pt", ".pth")
WEIGHTS_EXTENSIONS = (".pt", ".weights", ".h5")


def download_weights(weights_dir="model/weights", url=WEIGHTS_URL):
    print("grabbing model weights")
    return wget.download(url=url, out=weights_dir)


def download_trained_model(trained_dir="model/trained", url=TRAINED_MODEL_URL):
    print("grabbing trained model")
    return wget.download(url=url, out=trained_dir)


def find_file(directory, extensions):
    for file in os.listdir(directory):
        if file.endswith(extensions):
            return os.path.join(directory, file)
    return None


def locate_models(trained_dir="model/trained", weights_dir="model/weights"):
    trainedModel = find_file(trained_dir, TRAINED_EXTENSIONS)
    if trainedModel:
        print(f"Trained model file found: {trainedModel}")

    modelWeights = find_file(weights_dir, WEIGHTS_EXTENSIONS)
    if modelWeights:
        print(f"Weights file found: {modelWeights}")

    return trainedModel, modelWeights