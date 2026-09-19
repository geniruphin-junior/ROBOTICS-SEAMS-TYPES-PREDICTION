# import packages
from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from pathlib import Path
from architecture import VisionCNN
import sys
from io import BytesIO

# creat variables and dependacies
app = FastAPI(title="CNN API")
model_file = Path("Models/model 20260913_011137.pth")
folder = Path("models")
dataset_path = Path("dataset")
dataset_folders = [folder.name for folder in dataset_path.iterdir()]

# charge model (last  and best model)
try:
    if not model_file.exists():
        raise FileNotFoundError
    model = VisionCNN()
    model.load_state_dict(torch.load(model_file))
    model.eval()

except FileNotFoundError:
    sys.exit()


# show alls models
@app.get("/models")
def get_models():
    liste_models = []
    for version in folder.iterdir():
        liste_models.append(version.name)
    return {"Modeles": f"{liste_models}"}


# show parameters
@app.get("/parameters")
def get_parameters():
    n_parameters = sum(p.numel() for p in model.parameters())
    biais = []

    return {"nombres de parametres ": n_parameters, "bias": biais}


@app.post("/predict")
def predict(File: UploadFile(File(...))):
    File = Image.open(BytesIO(File))
    transform = transforms.Compile([transforms.Resize((96, 96)), transforms.Totensor()])
    File = transform(File)

    with torch.no_grad():
        probs = model(File)
        pred = torch.softmax(probs)
        proba = probs[pred]

    return {
        "prediction": f"cette image appartient au type de soudure {dataset_folders[int(pred.numpy())]}",
        "precision": proba * 100,
    }
