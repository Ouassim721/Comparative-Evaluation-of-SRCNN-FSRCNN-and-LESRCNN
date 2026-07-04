
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


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=9 // 2),
            nn.ReLU(inplace=True)
        )
        self.map = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2),
            nn.ReLU(inplace=True)
        )
        self.reconstruction = nn.Conv2d(32, 1, kernel_size=5, padding=5 // 2)

    def forward(self, x):
        x = self.features(x)
        x = self.map(x)
        x = self.reconstruction(x)
        return x


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


class LESRCNN(nn.Module):
    def __init__(self, scale):
        super(LESRCNN, self).__init__()
        self.ieeb = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
        )
        self.rb = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True),
        )
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


def load_srcnn(weight_path, device):
    model = SRCNN().to(device)
    if not os.path.isfile(weight_path):
        return None
    checkpoint = torch.load(weight_path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
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


def load_lesrcnn(weight_path, scale, device):
    import sys
    import os

    lesrcnn_dir = os.path.abspath(os.path.join("LESRCNN", "lesrcnn_b"))
    if lesrcnn_dir not in sys.path:
        sys.path.insert(0, lesrcnn_dir)

    try:
        from model.lesrcnn import Net
    except ImportError as e:
        print(f"  [ERREUR] Impossible d'importer model.lesrcnn : {e}")
        return None

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
        model.load_state_dict(cleaned_state, strict=False)
        model.eval()
        return model

# ===========================================================
#  PRÉ/POST-TRAITEMENT D'IMAGE
# ===========================================================


def bgr2ycbcr_y(img_bgr):
    img = img_bgr.astype(np.float32)
    y = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    y = y / 255.0
    return y.astype(np.float32)


def make_lr(img_hr, scale):
    h, w = img_hr.shape[:2]
    lr_h, lr_w = h // scale, w // scale
    img_lr = cv2.resize(img_hr, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
    img_bic = cv2.resize(img_lr, (w, h), interpolation=cv2.INTER_CUBIC)
    return img_lr, img_bic


def preprocess_srcnn(img_bic_y):
    t = torch.from_numpy(img_bic_y).unsqueeze(0).unsqueeze(0)
    return t.float()


def preprocess_fsrcnn(img_lr_y):
    t = torch.from_numpy(img_lr_y).unsqueeze(0).unsqueeze(0)
    return t.float()


def preprocess_lesrcnn(img_lr_bgr):
    img_rgb = cv2.cvtColor(
        img_lr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0)
    return t.float()


def tensor_to_image(tensor):
    arr = tensor.squeeze().clamp(0, 1).cpu().numpy()
    if arr.ndim == 3:
        arr = arr.transpose(1, 2, 0)
    return (arr * 255).astype(np.uint8)

# ===========================================================
#  ÉVALUATION D'UN MODÈLE SUR UNE IMAGE
# ===========================================================


def evaluate_model(model_name, model, img_hr_bgr, scale, device):
    if model is None:
        return None, None, None, None

    img_hr_y = bgr2ycbcr_y(img_hr_bgr)
    img_lr, img_bic = make_lr(img_hr_bgr, scale)
    img_lr_y = bgr2ycbcr_y(img_lr)
    img_bic_y = bgr2ycbcr_y(img_bic)

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
            else:
                inp = preprocess_lesrcnn(img_lr).to(device)
                start = time.perf_counter()
                out = model(inp, scale)
                end = time.perf_counter()

        inference_ms = (end - start) * 1000

        # --- CORRECTION DE LA NORMALISATION ET COULEURS ---
        if model_name == "LESRCNN":
            sr_rgb = (out.squeeze().clamp(0, 1).cpu().numpy(
            ).transpose(1, 2, 0) * 255).astype(np.uint8)
            sr_bgr = cv2.cvtColor(sr_rgb, cv2.COLOR_RGB2BGR)
            sr_y = (bgr2ycbcr_y(sr_bgr) * 255).astype(np.uint8)
        else:
            sr_y = (out.squeeze().clamp(0, 1).cpu().numpy()
                    * 255).astype(np.uint8)

            img_bic_ycrcb = cv2.cvtColor(img_bic, cv2.COLOR_BGR2YCrCb)

            min_h_color = min(sr_y.shape[0], img_bic_ycrcb.shape[0])
            min_w_color = min(sr_y.shape[1], img_bic_ycrcb.shape[1])
            img_bic_ycrcb = img_bic_ycrcb[:min_h_color, :min_w_color]

            img_bic_ycrcb[:, :, 0] = sr_y[:min_h_color, :min_w_color]

            sr_bgr = cv2.cvtColor(img_bic_ycrcb, cv2.COLOR_YCrCb2BGR)

        if model_name == "LESRCNN":
            sr_bgr = sr_bgr[:sr_y.shape[0], :sr_y.shape[1]]

        hr_ref_y = (img_hr_y * 255).astype(np.uint8)

        min_h = min(sr_y.shape[0], hr_ref_y.shape[0])
        min_w = min(sr_y.shape[1], hr_ref_y.shape[1])
        sr_y = sr_y[:min_h, :min_w]
        hr_ref_y = hr_ref_y[:min_h, :min_w]
        sr_bgr = sr_bgr[:min_h, :min_w]

        psnr = compute_psnr(hr_ref_y, sr_y, data_range=255)
        ssim = compute_ssim(hr_ref_y, sr_y, data_range=255)

        return round(psnr, 4), round(ssim, 4), round(inference_ms, 2), sr_bgr

    except Exception as e:
        print(f"    [ERREUR] {model_name} : {e}")
        return None, None, None, None


def evaluate_bicubic(img_hr_bgr, scale):
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Évaluation Super-Résolution CNN")
    print(f"  Device : {device}")
    print(f"{'='*60}\n")

    find_fsrcnn_weights()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for m in ["SRCNN", "FSRCNN", "LESRCNN", "Bicubic"]:
        os.makedirs(os.path.join(OUTPUT_DIR, m), exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, "resultats.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["Dataset", "Image", "Scale",
                    "Modele", "PSNR", "SSIM", "Temps_ms"])

    all_results = []

    for scale in SCALES:
        print(f"\n{'─'*50}")
        print(f"  SCALE x{scale}")
        print(f"{'─'*50}")

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

                bic_psnr, bic_ssim = evaluate_bicubic(img_hr, scale)
                psnr_accu["Bicubic"].append(bic_psnr)
                ssim_accu["Bicubic"].append(bic_ssim)
                writer.writerow([dataset_name, img_name, scale,
                                "Bicubic", bic_psnr, bic_ssim, "-"])
                print(f"Bic={bic_psnr:.2f}", end=" | ")

                for model_name, model in models.items():
                    psnr, ssim, t_ms, sr_img = evaluate_model(
                        model_name, model, img_hr, scale, device)

                    if psnr is not None:
                        psnr_accu[model_name].append(psnr)
                        ssim_accu[model_name].append(ssim)
                        writer.writerow(
                            [dataset_name, img_name, scale, model_name, psnr, ssim, t_ms])
                        print(f"{model_name}={psnr:.2f}", end=" | ")

                        out_path = os.path.join(
                            OUTPUT_DIR, model_name, f"{dataset_name}_{img_name}_x{scale}.png")
                        if sr_img is not None:
                            cv2.imwrite(out_path, sr_img)
                    else:
                        print(f"{model_name}=N/A", end=" | ")

                print()

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
