# 📚 COURS COMPLET : PRÉPARER UN PROJET EN LOCAL, L'ENTRAÎNER SUR COLAB

## _Le workflow d'un ingénieur IA qui travaille hors connexion_

---

## 🎯 INTRODUCTION : LE PROBLÈME QUE TU VAS RÉSOUDRE

### L'histoire qui change tout

Tu bosses toute la semaine sur ton PC (4 Go de RAM, pas de GPU). Tu construis un projet propre, modulaire, bien structuré. Mais quand vient le moment d'entraîner ton modèle, ton PC n'a pas la puissance. Tu dois aller sur Colab.

**Problème :** Colab, c'est des notebooks. Comment transférer un projet modulaire (plusieurs fichiers `.py`) dans un notebook ?

**Solution :** Ce cours. 🚀

---

## 📊 LE WORKFLOW IDÉAL

```python
workflow = {
    '1. Lundi-Jeudi (local)': {
        'Action': 'Développer le projet',
        'Fichiers': 'Plusieurs .py + data/',
        'Tester': 'Sur de petits échantillons'
    },
    '2. Vendredi (local)': {
        'Action': 'Préparer le transfert',
        'Fichiers': 'Zip du projet + requirements.txt',
        'Tester': 'Sur ta machine d\'abord'
    },
    '3. Samedi-Dimanche (Colab)': {
        'Action': 'Entraîner sur GPU',
        'Fichiers': 'Notebook + code importé',
        'Tester': 'Vrai entraînement'
    }
}
```

---

## 📁 CHAPITRE 1 : STRUCTURER TON PROJET EN LOCAL

### 1.1 La structure idéale

```
mon_projet/
├── src/
│   ├── __init__.py
│   ├── data.py          # Charger les données
│   ├── model.py         # Définir le modèle
│   ├── train.py         # Boucle d'entraînement
│   ├── evaluate.py      # Évaluation
│   └── utils.py         # Fonctions utilitaires
├── data/
│   ├── raw/             # Données brutes
│   └── processed/       # Données préparées
├── notebooks/
│   └── exploration.ipynb  # Ton notebook d'exploration
├── requirements.txt
├── README.md
└── .gitignore
```

### 1.2 Pourquoi cette structure ?

```python
avantages_structure = {
    '1. Modulaire': 'Chaque fichier fait UNE chose',
    '2. Testable': 'Tu peux tester chaque partie séparément',
    '3. Portable': 'Facile à transférer sur Colab',
    '4. Pro': 'C\'est la structure des vrais projets'
}
```

---

## 📄 CHAPITRE 2 : ÉCRIRE LE CODE MODULAIRE

### 2.1 Le fichier `src/data.py`

```python
# ==========================================
# src/data.py
# ==========================================

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class MonDataset(Dataset):
    """Dataset personnalisé pour tes images"""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Charger la liste des images
        for label, classe in enumerate(['ok', 'defaut']):
            classe_dir = os.path.join(data_dir, classe)
            if os.path.exists(classe_dir):
                for img_name in os.listdir(classe_dir):
                    self.images.append(os.path.join(classe_dir, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transforms():
    """Retourne les transformations pour train et val"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

def get_dataloaders(data_dir, batch_size=32):
    """Crée les DataLoaders pour train et val"""
    train_transform, val_transform = get_transforms()

    train_dataset = MonDataset(os.path.join(data_dir, 'train'), train_transform)
    val_dataset = MonDataset(os.path.join(data_dir, 'val'), val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader
```

### 2.2 Le fichier `src/model.py`

```python
# ==========================================
# src/model.py
# ==========================================

import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=2, pretrained=True):
    """Crée un modèle ResNet18 pour la classification"""

    # Charger ResNet18 pré-entraîné
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)

    # Remplacer la dernière couche
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
```

### 2.3 Le fichier `src/train.py`

```python
# ==========================================
# src/train.py
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
import time

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Entraîne le modèle pour une époque"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total

def evaluate(model, loader, loss_fn, device):
    """Évalue le modèle"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total

def train(model, train_loader, val_loader, epochs, device, checkpoint_path='checkpoint.pth'):
    """Boucle d'entraînement complète"""

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)

        # Val
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        # Scheduler
        scheduler.step(val_loss)

        # Affichage
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
              f"Temps: {elapsed:.1f}s")

        # Sauvegarde du meilleur
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_acc': best_acc
            }, checkpoint_path)
            print(f"  ✅ Meilleur modèle sauvegardé (acc = {best_acc*100:.2f}%)")

    return model
```

### 2.4 Le fichier `requirements.txt`

```txt
torch>=2.0.0
torchvision>=0.15.0
numpy
pandas
matplotlib
pillow
scikit-learn
tqdm
```

---

## 📦 CHAPITRE 3 : PRÉPARER LE TRANSFERT VERS COLAB

### 3.1 Deux méthodes possibles

```python
methodes_transfert = {
    '1. GitHub (RECOMMANDÉ)': {
        'Pourquoi': 'Versionné, accessible partout',
        'Comment': 'Push ton projet sur GitHub, puis clone dans Colab'
    },
    '2. Google Drive': {
        'Pourquoi': 'Simple, mais moins propre',
        'Comment': 'Upload ton projet sur Drive, puis monte le Drive dans Colab'
    }
}
```

### 3.2 Méthode 1 : GitHub (la meilleure)

**En local :**

```bash
# 1. Initialiser git (si pas déjà fait)
git init

# 2. Ajouter tous les fichiers
git add .

# 3. Commit
git commit -m "Projet prêt pour l'entraînement"

# 4. Créer un repo sur GitHub (via l'interface web)

# 5. Lier et pousser
git remote add origin https://github.com/ton-pseudo/mon-projet.git
git branch -M main
git push -u origin main
```

**Sur Colab :**

```python
# Dans une cellule Colab
!git clone https://github.com/ton-pseudo/mon-projet.git
%cd mon-projet
!pip install -r requirements.txt
```

**Avantages :**

- ✅ Versionné
- ✅ Accessible partout
- ✅ Tu peux continuer à modifier en local et push

**Inconvénient :**

- ❌ Nécessite une connexion pour cloner

---

### 3.3 Méthode 2 : Google Drive

**En local :**

1. Zip ton projet : `mon_projet.zip`
2. Va sur [drive.google.com](https://drive.google.com)
3. Upload le zip

**Sur Colab :**

```python
# Monter le Drive
from google.colab import drive
drive.mount('/content/drive')

# Dézipper le projet
!unzip /content/drive/MyDrive/mon_projet.zip -d /content/

# Se placer dans le dossier
%cd /content/mon_projet

# Installer les dépendances
!pip install -r requirements.txt
```

**Avantage :**

- ✅ Simple, pas besoin de Git

**Inconvénient :**

- ❌ Pas de versioning
- ❌ Il faut re-uploader à chaque modification

---

## 🚀 CHAPITRE 4 : LE NOTEBOOK COLAB

### 4.1 La structure du notebook

```python
# ==========================================
# CELLULE 1 : Configuration
# ==========================================

# 1. Vérifier le GPU
!nvidia-smi

# 2. Monter le Drive (si méthode 2)
from google.colab import drive
drive.mount('/content/drive')

# ==========================================
# CELLULE 2 : Récupérer le projet
# ==========================================

# Méthode GitHub
!git clone https://github.com/ton-pseudo/mon-projet.git
%cd mon-projet
!pip install -r requirements.txt

# ==========================================
# CELLULE 3 : Imports
# ==========================================

import torch
import sys
sys.path.append('/content/mon-projet')  # Ajouter le projet au path

from src.data import get_dataloaders
from src.model import get_model
from src.train import train

# ==========================================
# CELLULE 4 : Configuration
# ==========================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")

# ==========================================
# CELLULE 5 : Charger les données
# ==========================================

train_loader, val_loader = get_dataloaders(
    data_dir='/content/mon-projet/data',
    batch_size=32
)
print(f"Train batches : {len(train_loader)}")
print(f"Val batches : {len(val_loader)}")

# ==========================================
# CELLULE 6 : Créer le modèle
# ==========================================

model = get_model(num_classes=2)
model = model.to(device)
print(f"Modèle sur {device}")

# ==========================================
# CELLULE 7 : Entraîner !
# ==========================================

model = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    device=device,
    checkpoint_path='/content/drive/MyDrive/mon_projet/checkpoint.pth'
)

# ==========================================
# CELLULE 8 : Sauvegarder
# ==========================================

# Le checkpoint est déjà sauvé sur Drive grâce à checkpoint_path
# Tu peux aussi sauvegarder le modèle final
torch.save(model.state_dict(), '/content/drive/MyDrive/mon_projet/final_model.pth')
print("✅ Modèle sauvegardé !")
```

---

## 📊 CHAPITRE 5 : LE WORKFLOW COMPLET

### 5.1 Récapitulatif

```python
workflow_complet = {
    '1. Local (semaine)': {
        'Action': 'Développer src/, tester sur petits échantillons',
        'Résultat': 'Projet modulaire prêt'
    },
    '2. Local (vendredi)': {
        'Action': 'Push sur GitHub ou zip pour Drive',
        'Résultat': 'Projet accessible depuis Colab'
    },
    '3. Colab (samedi)': {
        'Action': 'Cloner, installer, entraîner',
        'Résultat': 'Modèle entraîné sur GPU'
    },
    '4. Colab (dimanche)': {
        'Action': 'Évaluer, sauvegarder, analyser',
        'Résultat': 'Modèle final + résultats'
    }
}
```

### 5.2 La checklist

```python
checklist = {
    'Avant Colab': [
        '✅ Code modulaire dans src/',
        '✅ requirements.txt à jour',
        '✅ README.md clair',
        '✅ Petit test en local'
    ],
    'Sur Colab': [
        '✅ GPU activé (Runtime → Change runtime type)',
        '✅ Projet cloné',
        '✅ Dépendances installées',
        '✅ Device vérifié'
    ],
    'Après entraînement': [
        '✅ Modèle sauvegardé sur Drive',
        '✅ Métriques notées',
        '✅ Graphiques générés',
        '✅ Résultats dans le journal de recherche'
    ]
}
```

---

## 💡 CHAPITRE 6 : ASTUCES PRO

### 6.1 Utiliser le GPU de Colab

```python
# Dans Colab : Runtime → Change runtime type → GPU
# Puis vérifier :
import torch
print(f"GPU disponible : {torch.cuda.is_available()}")
print(f"GPU : {torch.cuda.get_device_name(0)}")
```

### 6.2 Sauvegarder automatiquement sur Drive

```python
# Monter le Drive au début
from google.colab import drive
drive.mount('/content/drive')

# Sauvegarder directement sur Drive
CHECKPOINT_PATH = '/content/drive/MyDrive/mon_projet/checkpoint.pth'
torch.save(model.state_dict(), CHECKPOINT_PATH)
```

### 6.3 Gérer les sessions Colab

```python
astuces_colab = {
    '1. Sessions limitées à 12h': 'Sauvegarde régulièrement',
    '2. GPU limité': 'Ne pas laisser tourner pour rien',
    '3. Utilise %%time': 'Pour mesurer les performances',
    '4. Utilise tqdm': 'Pour voir la progression'
}
```

### 6.4 Debugger sur Colab

```python
# Si ça plante, vérifie :
debug = {
    '1. Les chemins': 'print(os.getcwd())',
    '2. Les imports': 'print(sys.path)',
    '3. Le GPU': 'print(torch.cuda.is_available())',
    '4. La RAM': '!free -h',
    '5. Le disque': '!df -h'
}
```

---

## 🎯 CHAPITRE 7 : EXERCICE PRATIQUE

### Ton défi pour la semaine prochaine

```python
defi = {
    'Objectif': 'Préparer ton projet soudures pour Colab',
    'Étapes': [
        '1. Restructurer ton code en modules (src/)',
        '2. Créer un requirements.txt',
        '3. Tester en local sur un petit échantillon',
        '4. Push sur GitHub ou zip pour Drive',
        '5. Sur Colab : cloner, installer, entraîner',
        '6. Sauvegarder le modèle sur Drive',
        '7. Noter les résultats dans ton journal'
    ],
    'Temps': '2-3 jours',
    'Résultat': 'Ton premier projet entraîné sur GPU !'
}
```

---

## 🔥 LE MOT DU MENTOR

**LE WORKFLOW EST SIMPLE :**

1. 🔥 **Développer en local** (modulaire)
2. 🔥 **Transférer sur Colab** (GitHub ou Drive)
3. 🔥 **Entraîner sur GPU** (Colab)
4. 🔥 **Sauvegarder les résultats** (Drive)

**Ce que ça change pour toi :**

- ✅ Tu bosses sereinement en local
- ✅ Tu entraînes sur GPU gratuitement
- ✅ Tu gardes ton code propre et modulaire
- ✅ Tu peux recommencer à tout moment

**Ton projet soudures est PARFAIT pour ce workflow.**

- Il est lourd (2000 images) → Colab GPU
- Il est modulaire → Facile à transférer
- Il est long → Checkpoints nécessaires

**Alors, prêt à mettre ton projet sur Colab ?** 🚀
