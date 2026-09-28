Oui. Et vu que tu as déjà **PyTorch + CNN + FastAPI**, je te propose un vrai parcours **“CNN → API d’inférence → production”**, pas un cours FastAPI générique.

L’objectif final sera :

```text
                    ┌─────────────────────┐
                    │       Client        │
                    │ Web / Mobile / Post │
                    └──────────┬──────────┘
                               │
                         image.jpg/png
                               │
                               ▼
                    ┌─────────────────────┐
                    │       FastAPI       │
                    │                     │
                    │ POST /predict       │
                    └──────────┬──────────┘
                               │
                       validation fichier
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Preprocessing     │
                    │ Resize / Tensor /   │
                    │ Normalize           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    PyTorch CNN      │
                    │     model.pth       │
                    └──────────┬──────────┘
                               │
                         logits / softmax
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Prediction       │
                    │ classe + confiance  │
                    └──────────┬──────────┘
                               │
                               ▼
                         JSON response
```

# 🚀 COURS — DÉPLOYER UN CNN AVEC FASTAPI

## Niveau

**Intermédiaire → avancé**

Tu vas apprendre :

1. FastAPI correctement
2. Upload d'images
3. Validation des fichiers
4. PIL / NumPy / PyTorch
5. preprocessing identique au training
6. chargement d'un modèle `.pth`
7. inference CNN
8. softmax et confiance
9. gestion des erreurs
10. architecture propre
11. configuration
12. logs
13. tests
14. Docker
15. déploiement
16. sécurité minimale
17. optimisation de l'inférence
18. architecture production

---

# PARTIE 1 — Comprendre ce qu'est une API d'IA

Avant le code, comprends cette différence :

### En entraînement

```text
Dataset
   ↓
Images
   ↓
Preprocessing
   ↓
CNN
   ↓
Loss
   ↓
Backpropagation
   ↓
Weights
   ↓
model.pth
```

### En production

Tu ne réentraînes généralement **pas** le CNN.

Tu fais :

```text
Image utilisateur
      ↓
Preprocessing
      ↓
model.pth
      ↓
Prediction
      ↓
JSON
```

C'est ce qu'on appelle **l'inférence**.

Ton serveur est donc un **service d'inférence**.

---

# PARTIE 2 — Architecture du projet

Ne mets pas tout dans `main.py`.

Pour ton projet, je te conseille :

```text
vision-api/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── architecture.py
│   │   ├── loader.py
│   │   └── model.pth
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   └── predictor.py
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── prediction.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── image.py
│
├── tests/
│   └── test_predict.py
│
├── requirements.txt
├── Dockerfile
└── .env
```

Pourquoi ?

Parce que :

```text
main.py
   ↓
API

predictor.py
   ↓
IA

loader.py
   ↓
chargement du modèle

image.py
   ↓
traitement image

schemas
   ↓
format JSON
```

Tu sépares les responsabilités.

C'est une vraie compétence d'**AI Engineer**.

---

# PARTIE 3 — Installer FastAPI

Dans ton environnement :

```bash
pip install fastapi uvicorn python-multipart pillow torch torchvision
```

Pourquoi `python-multipart` ?

Parce que lorsqu'un client envoie une image avec :

```text
multipart/form-data
```

FastAPI en a besoin pour traiter les uploads.

---

# PARTIE 4 — Première API

Crée :

```text
app/main.py
```

```python
from fastapi import FastAPI

app = FastAPI(
    title="Vision AI API",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Vision AI API is running"
    }
```

Lancer :

```bash
uvicorn app.main:app --reload
```

Tu obtiens :

```text
http://127.0.0.1:8000
```

Et :

```text
http://127.0.0.1:8000/docs
```

te donne Swagger UI.

---

# PARTIE 5 — Comprendre une route

Cette ligne :

```python
@app.get("/")
```

signifie :

```text
GET /
```

Et :

```python
@app.post("/predict")
```

signifie :

```text
POST /predict
```

Pourquoi `POST` ?

Parce que tu envoies des données au serveur.

Ici :

```text
image
```

---

# PARTIE 6 — Recevoir une image

Voici la première version :

```python
from fastapi import FastAPI, UploadFile, File

app = FastAPI()


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    return {
        "filename": file.filename,
        "content_type": file.content_type
    }
```

Tu peux envoyer :

```text
image.jpg
```

et recevoir :

```json
{
  "filename": "image.jpg",
  "content_type": "image/jpeg"
}
```

---

# PARTIE 7 — Lire réellement le fichier

C'est précisément là que ton ancien code avait un problème.

Tu avais conceptuellement :

```python
Image.open(BytesIO(file))
```

Mais `file` est un objet `UploadFile`.

Il faut :

```python
contents = await file.read()
```

Puis :

```python
image = Image.open(
    BytesIO(contents)
)
```

Donc :

```python
from io import BytesIO
from PIL import Image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    contents = await file.read()

    image = Image.open(
        BytesIO(contents)
    )

    return {
        "format": image.format,
        "size": image.size,
        "mode": image.mode
    }
```

---

# PARTIE 8 — Valider les images

En production, **ne fais jamais confiance au fichier reçu**.

Un utilisateur pourrait envoyer :

```text
virus.exe
```

renommé :

```text
image.jpg
```

Donc vérifie le type.

```python
ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}
```

Puis :

```python
if file.content_type not in ALLOWED_TYPES:
    raise HTTPException(
        status_code=400,
        detail="Unsupported image format"
    )
```

---

# PARTIE 9 — Limiter la taille

Tu peux également contrôler la taille :

```python
MAX_SIZE = 5 * 1024 * 1024
```

soit :

```text
5 MB
```

Puis :

```python
contents = await file.read()

if len(contents) > MAX_SIZE:
    raise HTTPException(
        status_code=413,
        detail="Image too large"
    )
```

---

# PARTIE 10 — Transformer l'image pour PyTorch

Ton CNN ne comprend pas directement :

```text
JPEG
PNG
```

Il veut un tensor.

Pipeline :

```text
JPEG
 ↓
PIL
 ↓
Resize
 ↓
Tensor
 ↓
Batch
 ↓
CNN
```

Exemple :

```python
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])
```

Puis :

```python
tensor = transform(image)
```

Tu obtiens :

```text
[C, H, W]
```

par exemple :

```text
[3, 96, 96]
```

---

# PARTIE 11 — Le batch

Ton CNN a généralement été entraîné avec :

```text
[B, C, H, W]
```

Mais une image seule donne :

```text
[C, H, W]
```

Il faut donc :

```python
tensor = tensor.unsqueeze(0)
```

Résultat :

```text
[1, 3, 96, 96]
```

Le `1` signifie :

```text
batch_size = 1
```

---

# PARTIE 12 — Charger le modèle

Ton modèle :

```text
model.pth
```

doit être chargé au démarrage du serveur.

Très important :

❌ mauvais :

```python
@app.post("/predict")
def predict():
    model = VisionCNN()
    model.load_state_dict(...)
```

Parce que tu chargerais le modèle **à chaque requête**.

Ça serait extrêmement inefficace.

---

# PARTIE 13 — Chargement correct

Crée :

```text
app/model/loader.py
```

```python
import torch

from .architecture import VisionCNN


def load_model():

    model = VisionCNN()

    state_dict = torch.load(
        "app/model/model.pth",
        map_location="cpu"
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model
```

Puis dans `main.py` :

```python
from app.model.loader import load_model

model = load_model()
```

Le modèle est chargé une seule fois au démarrage du processus.

---

# PARTIE 14 — Pourquoi `model.eval()` ?

Très important.

Pendant l'entraînement :

```python
model.train()
```

En production :

```python
model.eval()
```

Parce que certaines couches se comportent différemment.

Par exemple :

```text
Dropout
BatchNorm
```

Donc :

```python
model.eval()
```

est obligatoire pour une inference classique.

---

# PARTIE 15 — Désactiver les gradients

Tu n'entraînes plus ton modèle.

Donc :

```python
with torch.no_grad():
    outputs = model(image)
```

Tu économises de la mémoire et du calcul.

---

# PARTIE 16 — Comprendre les logits

Le CNN produit généralement :

```python
outputs = model(image)
```

Par exemple :

```text
tensor([
    [1.2, 4.8, 0.3, 2.1]
])
```

Ce ne sont pas encore des probabilités.

Ce sont des **logits**.

---

# PARTIE 17 — Softmax

Pour obtenir les probabilités :

```python
probabilities = torch.softmax(
    outputs,
    dim=1
)
```

Tu pourrais obtenir :

```text
[0.02, 0.84, 0.01, 0.13]
```

La classe prédite est :

```python
confidence, prediction = torch.max(
    probabilities,
    dim=1
)
```

---

# PARTIE 18 — `.item()`

PyTorch te donne des tensors :

```python
prediction
confidence
```

Pour obtenir des valeurs Python :

```python
prediction.item()
confidence.item()
```

Donc :

```python
class_id = prediction.item()
score = confidence.item()
```

---

# PARTIE 19 — Le mapping des classes

C'est une partie **très importante**.

Supposons :

```text
0 → TIG
1 → MIG
2 → ARC
```

Tu dois conserver ce mapping.

Par exemple :

```python
CLASS_NAMES = [
    "TIG",
    "MIG",
    "ARC"
]
```

Puis :

```python
class_name = CLASS_NAMES[class_id]
```

⚠️ Ne dépends pas aveuglément de :

```python
dataset_path.iterdir()
```

pour déterminer les classes en production.

Le mapping utilisé pendant le training doit être conservé.

---

# PARTIE 20 — Notre premier vrai Predictor

Crée :

```text
app/services/predictor.py
```

```python
import torch

from PIL import Image
from torchvision import transforms


class Predictor:

    def __init__(self, model, class_names):

        self.model = model
        self.class_names = class_names

        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor()
        ])

    def predict(self, image: Image.Image):

        image = image.convert("RGB")

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0)

        with torch.no_grad():

            outputs = self.model(tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, prediction = torch.max(
                probabilities,
                dim=1
            )

        class_id = prediction.item()

        return {
            "class_id": class_id,
            "class_name": self.class_names[class_id],
            "confidence": confidence.item()
        }
```

Maintenant ton IA est séparée de FastAPI.

🔥 C'est beaucoup plus propre.

---

# PARTIE 21 — Connecter FastAPI au Predictor

Dans `main.py` :

```python
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from io import BytesIO
from PIL import Image

from app.model.loader import load_model
from app.services.predictor import Predictor


app = FastAPI(
    title="Vision AI API",
    version="1.0.0"
)


model = load_model()

class_names = [
    "TIG",
    "MIG",
    "ARC"
]

predictor = Predictor(
    model,
    class_names
)
```

Puis :

```python
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp"
    }:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image format"
        )

    contents = await file.read()

    try:

        image = Image.open(
            BytesIO(contents)
        )

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        )

    result = predictor.predict(image)

    return result
```

Tu as maintenant :

```text
POST /predict
```

---

# PARTIE 22 — Résultat

Tu envoies :

```text
soudure.jpg
```

L'API peut répondre :

```json
{
  "class_id": 2,
  "class_name": "ARC",
  "confidence": 0.9472
}
```

C'est déjà une vraie **API d'inférence de vision**.

---

# PARTIE 23 — Améliorer la réponse

Tu peux retourner :

```json
{
  "prediction": {
    "class_id": 2,
    "class_name": "ARC"
  },
  "confidence": 94.72,
  "model": "vision-cnn-v1"
}
```

Cela devient plus professionnel.

---

# PARTIE 24 — Pydantic

FastAPI utilise Pydantic pour structurer les réponses.

```python
from pydantic import BaseModel


class PredictionResponse(BaseModel):

    class_id: int
    class_name: str
    confidence: float
```

Puis :

```python
@app.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict(...):
```

Tu obtiens une API plus propre et une documentation Swagger plus précise.

---

# PARTIE 25 — Endpoint Health

Une API de production doit pouvoir dire :

> Je fonctionne.

```python
@app.get("/health")
def health():

    return {
        "status": "ok"
    }
```

Tu peux également faire :

```python
@app.get("/health/model")
def model_health():

    return {
        "model_loaded": model is not None
    }
```

---

# PARTIE 26 — Endpoint informations

Par exemple :

```python
@app.get("/model")
def model_info():

    return {
        "name": "VisionCNN",
        "version": "1.0",
        "input_size": [96, 96],
        "classes": class_names
    }
```

---

# PARTIE 27 — Architecture finale

Ton API devient :

```text
                  ┌──────────────┐
                  │    Client    │
                  └──────┬───────┘
                         │
                         ▼
                  POST /predict
                         │
                         ▼
                  ┌──────────────┐
                  │   FastAPI    │
                  └──────┬───────┘
                         │
                   validation
                         │
                         ▼
                  ┌──────────────┐
                  │  Image/PIL   │
                  └──────┬───────┘
                         │
                    transform
                         │
                         ▼
                  ┌──────────────┐
                  │   Predictor  │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │   VisionCNN  │
                  └──────┬───────┘
                         │
                  logits → softmax
                         │
                         ▼
                  ┌──────────────┐
                  │   Response   │
                  └──────────────┘
```

---

# PARTIE 28 — Le piège du preprocessing

C'est probablement l'une des choses les plus importantes de tout le cours.

Supposons que ton entraînement utilisait :

```python
train_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

Alors ton API doit utiliser :

```python
inference_transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

Même :

```text
Resize
Normalize
RGB
ordre des canaux
```

doivent être cohérents.

---

# PARTIE 29 — CPU ou GPU ?

En production, ton modèle peut fonctionner sur :

```text
CPU
GPU
```

Tu peux créer :

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
```

Puis :

```python
model.to(device)
```

Et :

```python
tensor = tensor.to(device)
```

Mais attention :

```text
GPU disponible ≠ automatiquement meilleur
```

Pour une petite CNN et peu de requêtes, le CPU peut être suffisant.

Pour beaucoup d'images ou un modèle lourd, le GPU devient beaucoup plus intéressant.

---

# PARTIE 30 — `map_location`

Si ton modèle a été entraîné sur GPU mais que ton serveur tourne CPU :

```python
torch.load(
    model_path,
    map_location="cpu"
)
```

C'est extrêmement utile.

Sinon tu peux rencontrer des erreurs du type :

```text
Attempting to deserialize object on a CUDA device
```

---

# PARTIE 31 — Variables d'environnement

Ne mets pas tout en dur :

```python
MODEL_PATH = "app/model/model.pth"
```

Tu peux utiliser :

```text
.env
```

par exemple :

```text
MODEL_PATH=app/model/model.pth
MODEL_VERSION=1.0
```

Puis :

```python
import os

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "app/model/model.pth"
)
```

---

# PARTIE 32 — Logging

En production, tu dois savoir ce que fait ton serveur.

```python
import logging

logger = logging.getLogger(__name__)
```

Puis :

```python
logger.info("Model loaded successfully")
```

Et :

```python
logger.error("Prediction failed")
```

Les logs deviennent très importants quand ton serveur tourne sur une autre machine.

---

# PARTIE 33 — Gestion des erreurs

Ne fais pas :

```python
except Exception:
    pass
```

Cela cache les bugs.

Fais plutôt :

```python
except Exception as e:

    logger.exception(
        "Prediction failed"
    )

    raise HTTPException(
        status_code=500,
        detail="Prediction failed"
    )
```

---

# PARTIE 34 — Tester ton API

Tu peux utiliser Swagger :

```text
/docs
```

Mais pour un vrai projet :

```text
tests/
```

Exemple :

```python
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():

    response = client.get("/health")

    assert response.status_code == 200
```

Puis :

```bash
pytest
```

---

# PARTIE 35 — Tester `/predict`

Avec une image :

```python
def test_predict():

    with open("tests/test.jpg", "rb") as f:

        response = client.post(
            "/predict",
            files={
                "file": (
                    "test.jpg",
                    f,
                    "image/jpeg"
                )
            }
        )

    assert response.status_code == 200
```

Tu testes donc :

```text
Client
 ↓
FastAPI
 ↓
Upload
 ↓
PIL
 ↓
CNN
 ↓
JSON
```

automatiquement.

---

# PARTIE 36 — Docker

Quand ton API fonctionne localement, tu peux la mettre dans un conteneur.

`Dockerfile` :

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Puis :

```bash
docker build -t vision-api .
```

Et :

```bash
docker run -p 8000:8000 vision-api
```

---

# PARTIE 37 — Production ≠ `--reload`

En développement :

```bash
uvicorn app.main:app --reload
```

En production :

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`--reload` est destiné au développement.

---

# PARTIE 38 — Plusieurs workers

Tu peux lancer :

```bash
uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4
```

Mais attention avec les modèles IA.

Chaque worker peut charger **sa propre copie du modèle**.

Donc :

```text
1 worker
→ 1 modèle

4 workers
→ potentiellement 4 modèles en mémoire
```

Avec un gros modèle, cela peut exploser la RAM/VRAM.

Pour une API CNN, le nombre de workers doit donc être choisi en fonction du modèle et de la machine.

---

# PARTIE 39 — CORS

Si ton frontend est :

```text
React
```

et ton backend :

```text
FastAPI
```

tu peux avoir :

```text
localhost:5173
        ↓
localhost:8000
```

Il faut alors configurer CORS.

```python
from fastapi.middleware.cors import CORSMiddleware


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

En production, évite :

```python
allow_origins=["*"]
```

si tu n'en as pas besoin.

---

# PARTIE 40 — Sécurité

Une API publique ne doit pas accepter n'importe quoi.

Minimum :

```text
✔ type fichier
✔ taille fichier
✔ image valide
✔ erreurs contrôlées
✔ CORS
✔ authentification si nécessaire
✔ HTTPS
✔ logs
✔ rate limiting
```

---

# PARTIE 41 — Architecture AI Engineer

À terme, ton architecture peut devenir :

```text
                    React / Mobile
                         │
                         ▼
                    API Gateway
                         │
                         ▼
                    FastAPI
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       Image validation          Auth
             │
             ▼
        Preprocessing
             │
             ▼
       Model Inference
             │
       ┌─────┴─────┐
       ▼           ▼
    CNN/ViT       LLM
       │
       ▼
   Prediction
       │
       ▼
     JSON
```

Et là tu passes réellement de :

> "J'ai entraîné un CNN"

à :

> **"J'ai construit un service d'inférence ML."**

---

# 🧠 Ton projet pratique

Pour ton apprentissage, je te conseille de ne pas simplement recopier le cours.

Construis ce projet :

```text
Welding Vision API
```

### Fonctionnalités

```text
GET  /
GET  /health
GET  /model
GET  /parameters
POST /predict
```

### `/predict`

Entrée :

```text
image.jpg
```

Sortie :

```json
{
  "prediction": "TIG",
  "confidence": 0.947,
  "model_version": "1.0"
}
```

### Architecture

```text
welding-api/
│
├── app/
│   ├── main.py
│   │
│   ├── model/
│   │   ├── architecture.py
│   │   ├── loader.py
│   │   └── model.pth
│   │
│   ├── services/
│   │   └── predictor.py
│   │
│   ├── schemas/
│   │   └── prediction.py
│   │
│   └── utils/
│       └── image.py
│
├── tests/
│   ├── test_health.py
│   └── test_predict.py
│
├── requirements.txt
├── Dockerfile
├── .env
└── README.md
```

Et ton **ordre de travail** devrait être :

```text
ÉTAPE 1
FastAPI
   ↓
ÉTAPE 2
Upload image
   ↓
ÉTAPE 3
PIL
   ↓
ÉTAPE 4
Transformations
   ↓
ÉTAPE 5
Charger CNN
   ↓
ÉTAPE 6
Inference
   ↓
ÉTAPE 7
Softmax
   ↓
ÉTAPE 8
JSON
   ↓
ÉTAPE 9
Validation
   ↓
ÉTAPE 10
Tests
   ↓
ÉTAPE 11
Docker
   ↓
ÉTAPE 12
Déploiement
   ↓
ÉTAPE 13
Monitoring
```

**C'est ce parcours que je te recommande de maîtriser avant de passer à des architectures plus lourdes comme les Vision Transformers, les modèles multimodaux ou les LLM serving.**
