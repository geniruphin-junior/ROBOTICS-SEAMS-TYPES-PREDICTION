Ton projet sur les soudures est un excellent terrain d'application, et c'est un problème classique où l'augmentation de données fait des merveilles. Pour lutter contre l'overfitting (quand ton modèle apprend par cœur), `torchvision.transforms` est ton arme la plus puissante.

Voici un cours complet pour maîtriser ce module, avec un focus sur les transformations géométriques et les stratégies anti-overfitting.

### 📚 Cours : Maîtriser `torchvision.transforms` pour l'IA

#### 1. Pourquoi l'augmentation de données ?

L'augmentation de données est la technique la plus efficace pour améliorer les performances d'un CNN, surtout avec un petit jeu de données. L'idée est simple : on applique des transformations aléatoires aux images d'entraînement pour créer de nouvelles variations plausibles.

Un chat retourné horizontalement reste un chat. En entraînant le réseau sur ces variations, il apprend des caractéristiques **invariantes** à ces transformations, ce qui améliore considérablement sa capacité de généralisation. C'est une forme de régularisation qui est souvent plus bénéfique que le dropout ou la weight decay.

#### 2. L'outil : `torchvision.transforms.v2`

La version moderne et recommandée de l'API est `torchvision.transforms.v2`. Le point d'entrée principal est `v2.Compose`, qui te permet de construire un pipeline de transformations.

```python
from torchvision.transforms import v2

# Exemple de pipeline
transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),  # Convertit en tenseur d'image
    v2.ToDtype(torch.float32, scale=True), # Convertit en float et normalise [0,1]
    # ... autres transformations
])
```

#### 3. Les Transformations Géométriques (Le Cœur du Cours)

Ces transformations modifient l'agencement spatial des pixels. Elles sont les plus couramment utilisées et devraient être incluses dans presque tous les pipelines d'entraînement de CNN.

Voici les plus importantes :

- **`v2.RandomHorizontalFlip(p=0.5)`** : Retourne l'image horizontalement avec une probabilité de 50%. C'est la plus universellement utile.
- **`v2.RandomRotation(degrees=15)`** : Effectue une rotation aléatoire de l'image selon un angle compris dans l'intervalle spécifié.
- **`v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5)`** : Applique une transformation affine aléatoire. C'est une transformation très puissante qui combine :
  - `degrees` : rotation
  - `translate` : translation (déplacement) horizontale et verticale
  - `scale` : zoom avant/arrière
  - `shear` : cisaillement
- **`v2.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.9, 1.1))`** : Recadre une zone aléatoire de l'image puis la redimensionne à la taille spécifiée. C'est une technique d'augmentation très efficace. L'astuce est de régler `scale` sur `(0.8, 1.0)` pour éviter de couper trop d'informations cruciales.
- **`v2.RandomPerspective(distortion_scale=0.2, p=0.5)`** : Applique une distorsion de perspective aléatoire.
- **`v2.RandomVerticalFlip(p=0.5)`** : Retourne l'image verticalement. Moins courant que le flip horizontal, il peut être utile dans certains contextes (ex: images médicales).

**Exemple de pipeline géométrique complet :**

```python
geometric_transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15),
    v2.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    v2.RandomPerspective(distortion_scale=0.2, p=0.5),
    v2.RandomResizedCrop(size=224, scale=(0.8, 1.0))
])
```

#### 4. Les Transformations Photométriques (Bonus)

Ces transformations modifient les valeurs des pixels sans changer leur position, enseignant au réseau l'invariance aux conditions d'éclairage.

- **`v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1)`** : Modifie aléatoirement la luminosité, le contraste, la saturation et la teinte.
- **`v2.RandomGrayscale(p=0.1)`** : Convertit aléatoirement l'image en niveaux de gris.
- **`v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))`** : Applique un flou gaussien aléatoire.

#### 5. La Stratégie Anti-Overfitting : Comment les utiliser ?

L'overfitting se manifeste quand la perte d'entraînement diminue mais que la perte de validation stagne ou remonte. L'augmentation de données est ton principal levier.

**Voici la stratégie gagnante en 3 points :**

1.  **Un pipeline pour l'entraînement, un autre pour la validation** : C'est le point le plus important. **Tu ne dois JAMAIS appliquer d'augmentation de données sur ton ensemble de validation ou de test**. La validation doit rester "propre" pour mesurer objectivement la performance.

    ```python
    # Pipeline pour l'ENTRAÎNEMENT (avec augmentation)
    train_transform = v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(15),
        # ... autres augmentations
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Pipeline pour la VALIDATION/TEST (sans augmentation)
    val_transform = v2.Compose([
        v2.Resize(256), # Redimensionnement standard
        v2.CenterCrop(224), # Recadrage central
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ```

2.  **Trouve le "sweet spot"** : Une augmentation excessive peut paradoxalement nuire aux performances. Une étude sur CIFAR-10 a montré qu'une stratégie d'augmentation de base (flip + crop) obtient de meilleurs résultats (79.04% de précision) qu'une stratégie minimale (77.15%) ou excessive (72.17%). **Commence simple, puis complexifie.**

3.  **Combinaison avec d'autres techniques** : L'augmentation de données est encore plus efficace lorsqu'elle est combinée avec d'autres techniques de régularisation :
    - **Dropout** : Désactive aléatoirement des neurones pendant l'entraînement.
    - **Weight Decay (L2)** : Pénalise les poids trop grands.
    - **Early Stopping** : Arrête l'entraînement quand la performance de validation ne s'améliore plus.

#### 🎯 Exercice Pratique

Pour ton projet de soudures, voici un défi : implémente deux pipelines de transformation (un pour l'entraînement, un pour la validation) dans ton script `data.py`. Utilise `RandomHorizontalFlip`, `RandomRotation` et `RandomResizedCrop` avec des paramètres modérés. Entraîne ton modèle et observe l'écart entre la perte d'entraînement et de validation. Tu devrais voir cet écart se réduire, signe que l'overfitting est mieux contrôlé.

N'hésite pas si tu as des questions sur l'implémentation dans ton projet !
