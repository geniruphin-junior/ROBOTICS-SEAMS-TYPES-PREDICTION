import random
import matplotlib.pyplot as plt


def show_random_images(dataset, n=8, cols=4, title="Aperçu aléatoire"):
    """
    Affiche n images tirées au hasard dans un dataset PyTorch.
    - dataset : objet qui supporte dataset[i] -> (img, label)
    - n       : nombre d'images à afficher
    - cols    : nombre de colonnes dans la grille
    """
    rows = (n + cols - 1) // cols  # arrondi supérieur
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    fig.suptitle(title, fontsize=14)

    # axes peut être un seul objet si 1x1 → on le met en liste
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    # tirage aléatoire SANS remise (indices uniques)
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))

    for ax, idx in zip(axes, indices):
        img, label = dataset[idx]

        # C×H×W → H×W×C pour matplotlib
        img = img.permute(1, 2, 0)

        # Si on a normalisé avec ToTensor après Normalize, il faut "dénormaliser"
        # (voir section 5). Ici on suppose juste ToTensor → pixels dans [0,1].

        ax.imshow(img)
        ax.set_title(f"idx={idx} | label={label}", fontsize=9)
        ax.axis("off")

    # cacher les axes vides
    for ax in axes[len(indices) :]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
