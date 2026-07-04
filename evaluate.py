"""
=============================================================
  Script d'évaluation unifié — Super-Résolution CNN
  Modèles : SRCNN, FSRCNN, LESRCNN
  Métriques : PSNR, SSIM, Temps d'inférence
=============================================================

STRUCTURE ATTENDUE DU PROJET :
project/
├── evaluate.py          ← CE FICHIER
├── SRCNN-PyTorch/       ← repo cloné
├── FSRCNN-PyTorch/      ← repo cloné
├── LESRCNN/             ← repo cloné
├── datasets/
│   ├── Set5/            ← images HR (butterfly.png, baby.png, etc.)
│   └── Set14/           ← images HR (baboon.png, comic.png, etc.)
└── results/             ← créé automatiquement par le script

POIDS ATTENDUS :
- SRCNN-PyTorch/results/pretrained_models/srcnn_x2-T91-7d6e0623.pth.tar
- SRCNN-PyTorch/results/pretrained_models/srcnn_x3-T91-919a959c.pth.tar
- SRCNN-PyTorch/results/pretrained_models/srcnn_x4-T91-7c460643.pth.tar
- FSRCNN-PyTorch/results/pretrained_models/fsrcnn_x2-T91-f791f07f.pth.tar
- FSRCNN-PyTorch/results/pretrained_models/fsrcnn_x3-T91-55ffd1d6.pth.tar"
- FSRCNN-PyTorch/results/pretrained_models/fsrcnn_x4-T91-97a30bfb.pth.tar"
- LESRCNN/x2/lesrcnn_x2.pth
- LESRCNN/x3/lesrcnn_x3.pth
- LESRCNN/x4/lesrcnn_x4.pth
"""

from skimage.metrics import structural_similarity as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import torch.nn as nn
import torch
import numpy as np
import cv2
import os
import sys
import time
import glob
import csv
import warnings
warnings.filterwarnings("ignore")


# ===========================================================
#                    CONFIGURATION
# ===========================================================

DATASETS = {
    "Set5":  "datasets/Set5",
    "Set14": "datasets/Set14",
}

SCALES = [2, 3, 4]

# Chemins vers les poids pré-entraînés
WEIGHTS = {
    "SRCNN": {
        2: "SRCNN-PyTorch/results/pretrained_models/srcnn_x2-T91-7d6e0623.pth.tar",
        3: "SRCNN-PyTorch/results/pretrained_models/srcnn_x3-T91-919a959c.pth.tar",
        4: "SRCNN-PyTorch/results/pretrained_models/srcnn_x4-T91-7c460643.pth.tar",
    },
    "FSRCNN": {
        2: "FSRCNN-PyTorch/results/pretrained_models/fsrcnn_x2-T91-f791f07f.pth.tar",
        3: "FSRCNN-PyTorch/results/pretrained_models/fsrcnn_x3-T91-55ffd1d6.pth.tar",
        4: "FSRCNN-PyTorch/results/pretrained_models/fsrcnn_x4-T91-97a30bfb.pth.tar",
    },
    "LESRCNN": {
        2: "LESRCNN/x2/lesrcnn_x2.pth",
        3: "LESRCNN/x3/lesrcnn_x3.pth",
        4: "LESRCNN/x4/lesrcnn_x4.pth",
    },
}

OUTPUT_DIR = "results"

# ===========================================================
#  DÉTECTION AUTOMATIQUE DES POIDS FSRCNN
# ===========================================================


def find_fsrcnn_weights():
    """Cherche automatiquement les fichiers .pth.tar de FSRCNN."""
    base = "FSRCNN-PyTorch/results/pretrained_models/"
    if not os.path.isdir(base):
        return
    for scale in [2, 3, 4]:
        pattern = os.path.join(base, f"fsrcnn_x{scale}*.pth.tar")
        matches = glob.glob(pattern)
        if matches:
            WEIGHTS["FSRCNN"][scale] = matches[0]
            print(f"  [AUTO] FSRCNN x{scale} → {matches[0]}")
        else:
            print(f"  [WARN] Poids FSRCNN x{scale} introuvable dans {base}")

# ===========================================================
#  DÉFINITIONS DES MODÈLES
# ===========================================================

# ---- SRCNN ----


"""
class SRCNN(nn.Module):
    #Reimplantation de SRCNN (Dong et al., 2014).

    def __init__(self):
        super(SRCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2),
        )

    def forward(self, x):
        return self.features(x)
"""

# ---- SRCNN (Version corrigée pour correspondre aux poids) ----


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # L'auteur a regroupé la première convolution dans un bloc "features"
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True)
        )
        # La deuxième convolution est dans un bloc "map"
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2),
            nn.ReLU(inplace=True)
        )
        # La dernière couche s'appelle "reconstruction"
        self.reconstruction = nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2)

    def forward(self, x):
        x = self.features(x)
        x = self.map(x)
        x = self.reconstruction(x)
        return x

# ---- FSRCNN ----


class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(num_channels, d, 5, 1, 2), nn.PReLU(d))
        self.shrink = nn.Sequential(nn.Conv2d(d, s, 1, 1, 0), nn.PReLU(s))
        self.map = nn.Sequential(
            nn.Conv2d(s, s, 3, 1, 1), nn.PReLU(s),
            nn.Conv2d(s, s, 3, 1, 1), nn.PReLU(s),
            nn.Conv2d(s, s, 3, 1, 1), nn.PReLU(s),
            nn.Conv2d(s, s, 3, 1, 1), nn.PReLU(s)
        )
        self.expand = nn.Sequential(nn.Conv2d(s, d, 1, 1, 0), nn.PReLU(d))
        self.deconv = nn.ConvTranspose2d(
            d, num_channels, 9, scale_factor, 4, output_padding=scale_factor-1)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = self.shrink(x)
        x = self.map(x)
        x = self.expand(x)
        x = self.deconv(x)
        return x


"""
# ---- FSRCNN ----
class FSRCNN(nn.Module):
    #Reimplantation de FSRCNN (Dong et al., 2016).

    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5 // 2),
            nn.PReLU(d),
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend(
                [nn.Conv2d(s, s, kernel_size=3, padding=3 // 2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(
            d, num_channels,
            kernel_size=9, stride=scale_factor,
            padding=9 // 2,
            output_padding=scale_factor - 1
        )

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x

"""
# ---- LESRCNN ----


class LESRCNN(nn.Module):
    """
    Reimplantation simplifiée de LESRCNN (Tian et al., 2020).
    Compatible Python 3 / PyTorch moderne.
    Charge les poids officiels depuis les fichiers .pth du repo original.
    """

    def __init__(self, scale):
        super(LESRCNN, self).__init__()
        # IEEB — Information Extraction and Enhancement Block
        self.ieeb = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
        )
        # RB — Reconstruction Block
        self.rb = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
        )
        # IRB — Information Refinement Block + sub-pixel upsampling
        self.irb = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, scale * scale, 3, 1, 1),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        residual = x
        out = self.ieeb(x)
        out = self.rb(out)
        out = self.irb(out)
        return out

# ===========================================================
#  CHARGEMENT DES POIDS
# ===========================================================


"""
def load_srcnn(weight_path, device):
    model = SRCNN().to(device)
    if not os.path.isfile(weight_path):
        print(f"  [ERREUR] Poids SRCNN introuvable : {weight_path}")
        return None
    checkpoint = torch.load(weight_path, map_location=device)
    # Le repo Lornatang stocke les poids dans checkpoint["state_dict"]
    state = checkpoint.get("state_dict", checkpoint)
    # Nettoyer les préfixes éventuels
    new_state = {}
    for k, v in state.items():
        new_key = k.replace("module.", "").replace("features.", "features.")
        new_state[new_key] = v
    try:
        model.load_state_dict(new_state, strict=False)
    except Exception as e:
        print(f"  [WARN] Chargement SRCNN partiel : {e}")
    model.eval()
    return model


def load_fsrcnn(weight_path, scale, device):
    model = FSRCNN(scale_factor=scale).to(device)
    if not os.path.isfile(weight_path):
        print(f"  [ERREUR] Poids FSRCNN introuvable : {weight_path}")
        return None
    checkpoint = torch.load(weight_path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    new_state = {k.replace("module.", ""): v for k, v in state.items()}
    try:
        model.load_state_dict(new_state, strict=False)
    except Exception as e:
        print(f"  [WARN] Chargement FSRCNN partiel : {e}")
    model.eval()
    return model
"""


def load_srcnn(weight_path, device):
    model = SRCNN().to(device)
    if not os.path.isfile(weight_path):
        return None
    checkpoint = torch.load(weight_path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state, strict=True)  # Strict est maintenant à True !
    model.eval()
    return model


def load_fsrcnn(weight_path, scale, device):
    model = FSRCNN(scale_factor=scale).to(device)
    if not os.path.isfile(weight_path):
        return None
    checkpoint = torch.load(weight_path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


"""
def load_lesrcnn(weight_path, scale, device):
    model = LESRCNN(scale=scale).to(device)
    if not os.path.isfile(weight_path):
        print(f"  [ERREUR] Poids LESRCNN introuvable : {weight_path}")
        return None
    # Les poids LESRCNN sont sauvegardés directement (pas de "state_dict" wrapping)
    state = torch.load(weight_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        model.load_state_dict(state, strict=False)
    except Exception as e:
        print(f"  [WARN] Chargement LESRCNN partiel : {e}")
    model.eval()
    return model
"""


def load_lesrcnn(weight_path, scale, device):
    import sys
    import os

    # 1. On ajoute le dossier LESRCNN au chemin Python pour pouvoir importer ses fichiers
    lesrcnn_dir = os.path.abspath("LESRCNN")
    if lesrcnn_dir not in sys.path:
        sys.path.insert(0, lesrcnn_dir)

    try:
        # 2. On importe la VRAIE classe LESRCNN depuis le fichier model.py de l'auteur
        from model import LESRCNN
    except ImportError as e:
        print(
            f"  [ERREUR] Impossible d'importer model.py depuis le dossier LESRCNN : {e}")
        return None

    # 3. On instancie le modèle officiel
    model = LESRCNN(scale=scale).to(device)

    if not os.path.isfile(weight_path):
        print(f"  [ERREUR] Poids introuvables : {weight_path}")
        return None

    checkpoint = torch.load(weight_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    # On nettoie le "module." au cas où (lié au multi-GPU de l'auteur)
    cleaned_state = {}
    for k, v in state.items():
        name = k.replace("module.", "")
        cleaned_state[name] = v

    try:
        # 4. On charge les poids strictement
        model.load_state_dict(cleaned_state, strict=True)
        model.eval()
        return model
    except RuntimeError as e:
        print(f"\n[ATTENTION LESRCNN] Discordance : {e}")
        return None


def load_lesrcnn(weight_path, scale, device):
    import sys
    import os

    # Ajouter le dossier lesrcnn_b au chemin pour les imports de l'auteur
    lesrcnn_dir = os.path.abspath(os.path.join("LESRCNN", "lesrcnn_b"))
    if lesrcnn_dir not in sys.path:
        sys.path.insert(0, lesrcnn_dir)

    try:
        from model.lesrcnn import Net
    except ImportError as e:
        print(f"  [ERREUR] Impossible d'importer model.lesrcnn : {e}")
        return None

    # On passe scale=scale pour que le bloc upsample se construise correctement !
    model = Net(scale=scale).to(device)

    if not os.path.isfile(weight_path):
        print(f"  [ERREUR] Poids LESRCNN introuvables : {weight_path}")
        return None

    checkpoint = torch.load(weight_path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)

    cleaned_state = {}
    for k, v in state.items():
        name = k.replace("module.", "")
        cleaned_state[name] = v

    try:
        model.load_state_dict(cleaned_state, strict=True)
        model.eval()
        return model
    except RuntimeError as e:
        # Secours non-strict si besoin
        model.load_state_dict(cleaned_state, strict=False)
        model.eval()
        return model
# ===========================================================
#  PRÉ/POST-TRAITEMENT D'IMAGE
# ===========================================================


def bgr2ycbcr_y(img_bgr):
    """Extrait le canal Y en espace YCbCr (standard SR, BT.601).

    img_bgr est dans l'ordre B, G, R (convention OpenCV).
    Formule standard (sur image en échelle [0,255]) :
        Y = 16 + (65.481*R + 128.553*G + 24.966*B) / 255
    On applique donc les coefficients dans l'ordre B, G, R pour
    matcher l'ordre réel des canaux du tableau numpy (sinon R et B
    sont intervertis, ce qui fausse Y pour tous les pixels colorés).
    """
    img = img_bgr.astype(np.float32)  # reste en échelle [0, 255]
    y = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    y = y / 255.0  # retour en [0, 1] pour le reste du pipeline
    return y.astype(np.float32)


def make_lr(img_hr, scale):
    """
    Génère une image LR depuis une image HR par downscaling bicubique,
    puis re-upscale à la taille HR (nécessaire pour SRCNN qui prend LR upscalée).
    """
    h, w = img_hr.shape[:2]
    lr_h, lr_w = h // scale, w // scale
    # Downscale
    img_lr = cv2.resize(img_hr, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
    # Re-upscale bicubique (input pour SRCNN)
    img_bic = cv2.resize(img_lr, (w, h), interpolation=cv2.INTER_CUBIC)
    return img_lr, img_bic


def preprocess_srcnn(img_bic_y):
    """Prépare le tenseur d'entrée pour SRCNN (prend l'image upscalée en canal Y)."""
    t = torch.from_numpy(img_bic_y).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    return t.float()


def preprocess_fsrcnn(img_lr_y):
    """Prépare le tenseur d'entrée pour FSRCNN (prend directement l'image LR)."""
    t = torch.from_numpy(img_lr_y).unsqueeze(0).unsqueeze(0)  # (1,1,H/s,W/s)
    return t.float()


"""
def preprocess_lesrcnn(img_lr_y):
    #Prépare le tenseur d'entrée pour LESRCNN (idem FSRCNN, sub-pixel interne).
    t = torch.from_numpy(img_lr_y).unsqueeze(0).unsqueeze(0)
    return t.float()


def tensor_to_image(tensor):
    #Convertit un tenseur (1,1,H,W) en np.ndarray uint8.
    arr = tensor.squeeze().clamp(0, 1).cpu().numpy()
    return (arr * 255).astype(np.uint8)
"""


def preprocess_lesrcnn(img_lr_bgr):
    """Prépare le tenseur d'entrée pour LESRCNN (image RGB, 3 canaux, normalisée [0, 1])."""
    # L'auteur utilise un traitement de moyenne RGB, on passe de BGR à RGB
    img_rgb = cv2.cvtColor(
        img_lr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(img_rgb.transpose(
        2, 0, 1)).unsqueeze(0)  # (1, 3, H, W)
    return t.float()


def tensor_to_image(tensor):
    """Convertit un tenseur (1,1,H,W) ou (1,3,H,W) en np.ndarray uint8."""
    arr = tensor.squeeze().clamp(0, 1).cpu().numpy()
    if arr.ndim == 3:  # Si le modèle a sorti 3 canaux (C, H, W) -> LESRCNN
        arr = arr.transpose(1, 2, 0)  # Reformatage en (H, W, C)
    return (arr * 255).astype(np.uint8)
# ===========================================================
#  ÉVALUATION D'UN MODÈLE SUR UNE IMAGE
# ===========================================================


def evaluate_model(model_name, model, img_hr_bgr, scale, device):
    """
    Lance l'inférence d'un modèle sur une image et retourne PSNR, SSIM, temps.
    """
    if model is None:
        return None, None, None, None

    # Préparer les images de base
    img_hr_y = bgr2ycbcr_y(img_hr_bgr)
    img_lr, img_bic = make_lr(img_hr_bgr, scale)
    img_lr_y = bgr2ycbcr_y(img_lr)
    img_bic_y = bgr2ycbcr_y(img_bic)

    # Crop pour que les dimensions soient divisibles par scale
    h, w = img_hr_y.shape
    h_crop = (h // scale) * scale
    w_crop = (w // scale) * scale
    img_hr_y = img_hr_y[:h_crop, :w_crop]
    img_bic_y = img_bic_y[:h_crop, :w_crop]

    try:
        with torch.no_grad():
            if model_name == "SRCNN":
                inp = preprocess_srcnn(img_bic_y).to(device)
                start = time.perf_counter()
                out = model(inp)
                end = time.perf_counter()
            elif model_name == "FSRCNN":
                inp = preprocess_fsrcnn(img_lr_y).to(device)
                start = time.perf_counter()
                out = model(inp)
                end = time.perf_counter()
            else:  # LESRCNN
                inp = preprocess_lesrcnn(img_lr).to(device)
                start = time.perf_counter()
                out = model(inp, scale)
                end = time.perf_counter()

        inference_ms = (end - start) * 1000

        # --- CORRECTION DE LA NORMALISATION ---
        if model_name == "LESRCNN":
            # La sortie de LESRCNN est un tenseur [0, 1] à 3 canaux RGB
            # 1. On le convertit en image numpy [0, 255] uint8 RGB
            sr_rgb = (out.squeeze().clamp(0, 1).cpu().numpy(
            ).transpose(1, 2, 0) * 255).astype(np.uint8)
            # 2. On passe en BGR pour OpenCV
            sr_bgr = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)
            # 3. On extrait le canal Y standard [0, 1] puis conversion en [0, 255]
            sr_y = (bgr2ycbcr_y(sr_bgr) * 255).astype(np.uint8)
        else:
            # SRCNN et FSRCNN sortent un canal unique Y dans l'échelle [0, 1]
            sr_y = (out.squeeze().clamp(0, 1).cpu().numpy()
                    * 255).astype(np.uint8)

        # Les références HR doivent aussi être strictement en [0, 255] uint8
        hr_ref_y = (img_hr_y * 255).astype(np.uint8)

        # Recadrer les dimensions pour qu'elles correspondent parfaitement
        min_h = min(sr_y.shape[0], hr_ref_y.shape[0])
        min_w = min(sr_y.shape[1], hr_ref_y.shape[1])
        sr_y = sr_y[:min_h, :min_w]
        hr_ref_y = hr_ref_y[:min_h, :min_w]

        # Calcul des métriques sur l'échelle standard 255
        psnr = compute_psnr(hr_ref_y, sr_y, data_range=255)
        ssim = compute_ssim(hr_ref_y, sr_y, data_range=255)

        # Pour la sauvegarde de l'image finale
        return round(psnr, 4), round(ssim, 4), round(inference_ms, 2), sr_y

    except Exception as e:
        print(f"    [ERREUR] {model_name} : {e}")
        return None, None, None, None

# ===========================================================
#  BICUBIQUE (BASELINE)
# ===========================================================


def evaluate_bicubic(img_hr_bgr, scale):
    """Baseline : interpolation bicubique simple."""
    img_hr_y = bgr2ycbcr_y(img_hr_bgr)
    h, w = img_hr_y.shape
    h_crop = (h // scale) * scale
    w_crop = (w // scale) * scale
    img_hr_y = img_hr_y[:h_crop, :w_crop]

    img_lr = cv2.resize(img_hr_bgr, (w_crop // scale, h_crop //
                        scale), interpolation=cv2.INTER_CUBIC)
    img_bic_bgr = cv2.resize(img_lr, (w_crop, h_crop),
                             interpolation=cv2.INTER_CUBIC)
    img_bic_y = bgr2ycbcr_y(img_bic_bgr)[:h_crop, :w_crop]

    hr_ref = (img_hr_y * 255).astype(np.uint8)
    bic_ref = (img_bic_y * 255).astype(np.uint8)

    psnr = compute_psnr(hr_ref, bic_ref, data_range=255)
    ssim = compute_ssim(hr_ref, bic_ref, data_range=255)
    return round(psnr, 4), round(ssim, 4)

# ===========================================================
#  PROGRAMME PRINCIPAL
# ===========================================================


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Évaluation Super-Résolution CNN")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    # Chercher automatiquement les poids FSRCNN
    find_fsrcnn_weights()

    # Créer les dossiers de sortie
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for m in ["SRCNN", "FSRCNN", "LESRCNN", "Bicubic"]:
        os.makedirs(os.path.join(OUTPUT_DIR, m), exist_ok=True)

    # Fichier CSV de résultats
    csv_path = os.path.join(OUTPUT_DIR, "resultats.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["Dataset", "Image", "Scale",
                    "Modele", "PSNR", "SSIM", "Temps_ms"])

    # Stocker les résultats pour affichage final
    all_results = []

    for scale in SCALES:
        print(f"\n{'─'*50}")
        print(f"  SCALE x{scale}")
        print(f"{'─'*50}")

        # Charger les modèles pour cette échelle
        models = {}
        print(f"\n  Chargement des modèles...")
        models["SRCNN"] = load_srcnn(WEIGHTS["SRCNN"][scale], device)
        models["FSRCNN"] = load_fsrcnn(WEIGHTS["FSRCNN"][scale], scale, device)
        models["LESRCNN"] = load_lesrcnn(
            WEIGHTS["LESRCNN"][scale], scale, device)

        for dataset_name, dataset_path in DATASETS.items():
            if not os.path.isdir(dataset_path):
                print(f"\n  [SKIP] Dataset introuvable : {dataset_path}")
                continue

            # Chercher toutes les images (PNG, BMP, JPG)
            image_files = (
                glob.glob(os.path.join(dataset_path, "*.png")) +
                glob.glob(os.path.join(dataset_path, "*.bmp")) +
                glob.glob(os.path.join(dataset_path, "*.jpg"))
            )

            if not image_files:
                print(f"\n  [SKIP] Aucune image trouvée dans : {dataset_path}")
                continue

            print(f"\n  Dataset : {dataset_name} ({len(image_files)} images)")

            psnr_accu = {m: []
                         for m in ["Bicubic", "SRCNN", "FSRCNN", "LESRCNN"]}
            ssim_accu = {m: []
                         for m in ["Bicubic", "SRCNN", "FSRCNN", "LESRCNN"]}

            for img_path in sorted(image_files):
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                img_hr = cv2.imread(img_path)
                if img_hr is None:
                    continue

                print(f"    {img_name:<20}", end=" | ")

                # Bicubique
                bic_psnr, bic_ssim = evaluate_bicubic(img_hr, scale)
                psnr_accu["Bicubic"].append(bic_psnr)
                ssim_accu["Bicubic"].append(bic_ssim)
                writer.writerow([dataset_name, img_name, scale,
                                "Bicubic", bic_psnr, bic_ssim, "-"])
                print(f"Bic={bic_psnr:.2f}", end=" | ")

                # Modèles CNN
                for model_name, model in models.items():
                    psnr, ssim, t_ms, sr_img = evaluate_model(
                        model_name, model, img_hr, scale, device)

                    if psnr is not None:
                        psnr_accu[model_name].append(psnr)
                        ssim_accu[model_name].append(ssim)
                        writer.writerow(
                            [dataset_name, img_name, scale, model_name, psnr, ssim, t_ms])
                        print(f"{model_name}={psnr:.2f}", end=" | ")

                        # Sauvegarder l'image SR
                        out_path = os.path.join(
                            OUTPUT_DIR, model_name, f"{dataset_name}_{img_name}_x{scale}.png")
                        if sr_img is not None:
                            cv2.imwrite(out_path, sr_img)
                    else:
                        print(f"{model_name}=N/A", end=" | ")

                print()

            # Afficher les moyennes par dataset/scale
            print(f"\n  ── Moyennes {dataset_name} x{scale} ──")
            print(f"  {'Modèle':<12} {'PSNR moy':>10} {'SSIM moy':>10}")
            print(f"  {'─'*35}")
            for m in ["Bicubic", "SRCNN", "FSRCNN", "LESRCNN"]:
                if psnr_accu[m]:
                    avg_p = round(np.mean(psnr_accu[m]), 4)
                    avg_s = round(np.mean(ssim_accu[m]), 4)
                    print(f"  {m:<12} {avg_p:>10.4f} {avg_s:>10.4f}")
                    all_results.append({
                        "dataset": dataset_name, "scale": scale,
                        "model": m, "psnr": avg_p, "ssim": avg_s
                    })

    csv_file.close()

    # ── Résumé final ──
    print(f"\n{'='*60}")
    print(f"  RÉSUMÉ FINAL")
    print(f"{'='*60}")
    print(f"\n  {'Dataset':<8} {'Scale':>6} {'Modèle':<12} {'PSNR':>8} {'SSIM':>8}")
    print(f"  {'─'*48}")
    for r in all_results:
        print(
            f"  {r['dataset']:<8} x{r['scale']:>5} {r['model']:<12} {r['psnr']:>8.4f} {r['ssim']:>8.4f}")

    print(f"\n  Résultats sauvegardés dans : {csv_path}")
    print(f"  Images SR dans             : {OUTPUT_DIR}/")
    print()


if __name__ == "__main__":
    main()
