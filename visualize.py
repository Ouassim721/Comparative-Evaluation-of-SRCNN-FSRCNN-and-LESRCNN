"""
=============================================================
  Script de visualisation comparative — Super-Résolution CNN
  Génère :
    1. Figures comparatives côte à côte (LR / SRCNN / FSRCNN / LESRCNN / HR)
    2. Graphiques PSNR et SSIM par modèle et dataset
    3. Tableau récapitulatif en image

  CORRECTIF (couleur LESRCNN) :
    Le modèle LESRCNN (via evaluate.py) ne reconstruit en réalité que
    la luminance (Y) et sauvegarde une image quasi grise (R≈G≈B), à la
    différence de SRCNN/FSRCNN dont la chrominance Bicubique est déjà
    réinjectée en amont. Ce script détecte automatiquement une image SR
    "grise" et lui réinjecte la chrominance (Cb/Cr) de l'image Bicubique
    correspondante, pour obtenir un rendu couleur cohérent avec les
    autres méthodes — sans avoir besoin de relancer evaluate.py.
=============================================================

UTILISATION (après avoir lancé evaluate.py) :
    python visualize.py
"""

from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import cv2
import os
import glob
import csv
import warnings
warnings.filterwarnings("ignore")

matplotlib.use("Agg")

# ===========================================================
#  CONFIGURATION
# ===========================================================

DATASETS = {
    "Set5":  "datasets/Set5",
    "Set14": "datasets/Set14",
}

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
CSV_PATH = "results/resultats.csv"

SCALE_DEMO = 2
MODELS = ["Bicubic", "SRCNN", "FSRCNN", "LESRCNN"]
COLORS = {
    "Bicubic": "#7F8C8D",
    "SRCNN":   "#C0392B",
    "FSRCNN":  "#2471A3",
    "LESRCNN": "#1E8449",
}

LIGHT_BG = "#FFFFFF"
LIGHT_PANEL = "#F4F4F4"
LIGHT_TEXT = "#1A1A1A"
LIGHT_GRID = "#CCCCCC"
LIGHT_SPINE = "#888888"
REF_COLOR = "#8E6E00"

# Seuil de saturation moyenne (HSV, 0-255) en dessous duquel une image
# RGB est considérée comme "grise" (canaux quasi identiques).
GRAY_SATURATION_THRESHOLD = 8.0

# ===========================================================
#  UTILITAIRES
# ===========================================================


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def bgr2ycbcr_y(img_bgr):
    img = img_bgr.astype(np.float32) / 255.0
    y = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    return np.clip(y, 0, 255).astype(np.uint8)


def make_lr_bicubic(img_hr, scale):
    h, w = img_hr.shape[:2]
    img_lr = cv2.resize(img_hr, (w // scale, h // scale),
                        interpolation=cv2.INTER_CUBIC)
    img_bic = cv2.resize(img_lr, (w, h), interpolation=cv2.INTER_CUBIC)
    return img_bic


def is_grayscale_like(img_rgb, sat_threshold=GRAY_SATURATION_THRESHOLD):
    """Détecte si une image RGB est en réalité quasi monochrome
    (R≈G≈B), typiquement quand un modèle n'a reconstruit que la
    luminance sans restaurer la chrominance."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mean_saturation = float(hsv[:, :, 1].mean())
    return mean_saturation < sat_threshold


def recolorize_with_reference_chroma(img_rgb, ref_rgb):
    """Réinjecte la chrominance (Cb/Cr) d'une image de référence couleur
    (ici la version Bicubique) dans la luminance fournie par le modèle SR,
    pour reconstituer une image couleur cohérente."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ref_bgr = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR)

    h = min(img_bgr.shape[0], ref_bgr.shape[0])
    w = min(img_bgr.shape[1], ref_bgr.shape[1])
    img_bgr = img_bgr[:h, :w]
    ref_bgr = ref_bgr[:h, :w]

    y_model = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ref_ycrcb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2YCrCb)
    ref_ycrcb[:, :, 0] = y_model
    recolored_bgr = cv2.cvtColor(ref_ycrcb, cv2.COLOR_YCrCb2BGR)
    return cv2.cvtColor(recolored_bgr, cv2.COLOR_BGR2RGB)


def load_csv(path):
    data = []
    if not os.path.isfile(path):
        print(f"[WARN] CSV introuvable : {path}")
        return data
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["PSNR"] = float(row["PSNR"])
                row["SSIM"] = float(row["SSIM"])
                row["Scale"] = int(row["Scale"])
            except:
                pass
            data.append(row)
    return data


def aggregate(data, dataset, scale):
    agg = {m: {"psnr": [], "ssim": []} for m in MODELS}
    for row in data:
        if row["Dataset"] == dataset and row["Scale"] == scale and row["Modele"] in MODELS:
            agg[row["Modele"]]["psnr"].append(row["PSNR"])
            agg[row["Modele"]]["ssim"].append(row["SSIM"])
    result = {}
    for m in MODELS:
        if agg[m]["psnr"]:
            result[m] = {
                "psnr": round(np.mean(agg[m]["psnr"]), 4),
                "ssim": round(np.mean(agg[m]["ssim"]), 4),
            }
    return result

# ===========================================================
#  FIGURE 1 — COMPARAISON CÔTE À CÔTE PAR IMAGE
# ===========================================================


def make_comparison_figure(img_name, dataset_name, scale):
    hr_path = None
    for ext in ["*.png", "*.bmp", "*.jpg"]:
        matches = glob.glob(os.path.join(DATASETS.get(dataset_name, ""), ext))
        for m in matches:
            if os.path.splitext(os.path.basename(m))[0].lower() == img_name.lower():
                hr_path = m
                break

    if hr_path is None:
        print(f"  [SKIP] Image HR introuvable pour {img_name}")
        return

    img_hr = cv2.imread(hr_path)
    if img_hr is None:
        return

    h, w = img_hr.shape[:2]
    h_crop = (h // scale) * scale
    w_crop = (w // scale) * scale
    img_hr = img_hr[:h_crop, :w_crop]

    img_bic = make_lr_bicubic(img_hr, scale)

    hr_rgb = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
    bic_rgb = cv2.cvtColor(img_bic, cv2.COLOR_BGR2RGB)

    # Images SR depuis results/ (CORRECTIF COULEUR ICI)
    sr_images = {}
    for model in ["SRCNN", "FSRCNN", "LESRCNN"]:
        sr_path = os.path.join(
            RESULTS_DIR, model, f"{dataset_name}_{img_name}_x{scale}.png")
        if os.path.isfile(sr_path):
            img_sr_bgr = cv2.imread(sr_path, cv2.IMREAD_COLOR)
            img_sr_rgb = cv2.cvtColor(img_sr_bgr, cv2.COLOR_BGR2RGB)

            if is_grayscale_like(img_sr_rgb):
                # Cas typique de LESRCNN : luminance seule -> on réinjecte
                # la chrominance de la Bicubique pour obtenir une image couleur.
                img_sr_rgb = recolorize_with_reference_chroma(
                    img_sr_rgb, bic_rgb)
                print(
                    f"    [FIX] Image {model} détectée en niveaux de gris -> recolorée ({img_name})")

            sr_images[model] = img_sr_rgb
        else:
            sr_images[model] = None

    hr_y = bgr2ycbcr_y(img_hr)
    bic_y = bgr2ycbcr_y(img_bic)

    panels = [
        ("LR (Bicubic)", bic_rgb),
        ("SRCNN",        sr_images.get("SRCNN")),
        ("FSRCNN",       sr_images.get("FSRCNN")),
        ("LESRCNN",      sr_images.get("LESRCNN")),
        ("HR Original",  hr_rgb),
    ]

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.5))
    fig.patch.set_facecolor(LIGHT_BG)

    from skimage.metrics import peak_signal_noise_ratio as calc_psnr
    from skimage.metrics import structural_similarity as calc_ssim

    for ax, (title, img) in zip(axes, panels):
        ax.set_facecolor(LIGHT_PANEL)
        if img is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    color=LIGHT_TEXT, fontsize=14, transform=ax.transAxes)
            ax.set_title(title, color=LIGHT_TEXT, fontsize=11, pad=8)
        else:
            ax.imshow(img)

            if title not in ("HR Original", "LR (Bicubic)"):
                img_bgr_tmp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_y_tmp = bgr2ycbcr_y(img_bgr_tmp)

                p = calc_psnr(
                    hr_y, img_y_tmp[:hr_y.shape[0], :hr_y.shape[1]], data_range=255)
                s = calc_ssim(
                    hr_y, img_y_tmp[:hr_y.shape[0], :hr_y.shape[1]], data_range=255)
                subtitle = f"PSNR={p:.2f} dB\nSSIM={s:.4f}"
                color = COLORS.get(title, LIGHT_TEXT)
            elif title == "LR (Bicubic)":
                p = calc_psnr(hr_y, bic_y, data_range=255)
                s = calc_ssim(hr_y, bic_y, data_range=255)
                subtitle = f"PSNR={p:.2f} dB\nSSIM={s:.4f}"
                color = COLORS["Bicubic"]
            else:
                subtitle = "Référence"
                color = REF_COLOR

            ax.set_title(title, color=color, fontsize=11,
                         fontweight="bold", pad=6)
            ax.text(0.5, -0.08, subtitle, ha="center", va="top",
                    transform=ax.transAxes, color=color, fontsize=8.5)

        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(f"Comparaison SR — {img_name} (Scale ×{scale})",
                 color=LIGHT_TEXT, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(
        FIGURES_DIR, f"comparison_{dataset_name}_{img_name}_x{scale}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Figure sauvegardée : {out_path}")


def make_bar_charts(data):
    scales = [2, 3, 4]
    dataset_names = list(DATASETS.keys())
    for dataset in dataset_names:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(LIGHT_BG)
        fig.suptitle(
            f"PSNR & SSIM moyen — {dataset}", color=LIGHT_TEXT, fontsize=14, fontweight="bold", y=1.01)
        x = np.arange(len(scales))
        bar_w = 0.18
        offsets = np.linspace(-(len(MODELS)-1)/2,
                              (len(MODELS)-1)/2, len(MODELS)) * bar_w
        for ax_idx, metric in enumerate(["psnr", "ssim"]):
            ax = axes[ax_idx]
            ax.set_facecolor(LIGHT_PANEL)
            for i, model in enumerate(MODELS):
                values = []
                for scale in scales:
                    agg = aggregate(data, dataset, scale)
                    values.append(agg.get(model, {}).get(metric, 0))
                bars = ax.bar(x + offsets[i], values, bar_w, label=model,
                              color=COLORS[model], alpha=0.92, edgecolor=LIGHT_TEXT, linewidth=0.4)
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (0.05 if metric == "psnr" else 0.001),
                                f"{val:.2f}" if metric == "psnr" else f"{val:.3f}", ha="center", va="bottom", color=LIGHT_TEXT, fontsize=7, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([f"×{s}" for s in scales],
                               color=LIGHT_TEXT, fontsize=11)
            ax.set_ylabel("PSNR (dB)" if metric ==
                          "psnr" else "SSIM", color=LIGHT_TEXT, fontsize=11)
            ax.set_title("PSNR moyen" if metric == "psnr" else "SSIM moyen",
                         color=LIGHT_TEXT, fontsize=12, pad=10)
            ax.tick_params(colors=LIGHT_TEXT)
            ax.spines[:].set_color(LIGHT_SPINE)
            ax.set_ylim(bottom=0)
            ax.legend(facecolor=LIGHT_BG, edgecolor=LIGHT_SPINE,
                      labelcolor=LIGHT_TEXT, fontsize=9)
            ax.grid(axis="y", color=LIGHT_GRID, linestyle="--", alpha=0.7)
        plt.tight_layout()
        out_path = os.path.join(FIGURES_DIR, f"barchart_{dataset}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  ✓ Graphique barres : {out_path}")


def make_summary_table(data):
    rows = []
    for dataset in DATASETS.keys():
        for scale in [2, 3, 4]:
            agg = aggregate(data, dataset, scale)
            for model in MODELS:
                if model in agg:
                    rows.append(
                        [dataset, f"×{scale}", model, f"{agg[model]['psnr']:.4f}", f"{agg[model]['ssim']:.4f}"])
    if not rows:
        print("  [SKIP] Aucune donnée pour le tableau récapitulatif.")
        return
    headers = ["Dataset", "Scale", "Modèle", "PSNR moy (dB)", "SSIM moy"]
    n_rows = len(rows) + 1
    n_cols = len(headers)
    fig_h = max(5, 0.4 * n_rows + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)
    ax.axis("off")
    col_widths = [0.14, 0.10, 0.15, 0.22, 0.18]
    x_positions = [sum(col_widths[:i]) for i in range(n_cols)]
    total_w = sum(col_widths)
    row_h = 0.88 / n_rows
    for j, (header, x) in enumerate(zip(headers, x_positions)):
        ax.text(x / total_w + col_widths[j] / total_w / 2, 1 - row_h / 2, header, ha="center",
                va="center", transform=ax.transAxes, color=LIGHT_TEXT, fontsize=10, fontweight="bold")
    for i, row in enumerate(rows):
        y = 1 - row_h * (i + 1.5)
        bg_color = LIGHT_PANEL if i % 2 == 0 else LIGHT_BG
        model_color = COLORS.get(row[2], LIGHT_TEXT)
        rect = FancyBboxPatch((0, y - row_h * 0.45), 1, row_h * 0.9, boxstyle="round,pad=0.005",
                              facecolor=bg_color, edgecolor="none", transform=ax.transAxes, zorder=0)
        ax.add_patch(rect)
        for j, (cell, x) in enumerate(zip(row, x_positions)):
            color = model_color if j == 2 else LIGHT_TEXT
            fw = "bold" if j in (2, 3) else "normal"
            ax.text(x / total_w + col_widths[j] / total_w / 2, y, cell, ha="center",
                    va="center", transform=ax.transAxes, color=color, fontsize=9, fontweight=fw)
    ax.set_title("Résultats comparatifs — Super-Résolution CNN",
                 color=LIGHT_TEXT, fontsize=13, fontweight="bold", pad=12)
    out_path = os.path.join(FIGURES_DIR, "summary_table.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Tableau récapitulatif : {out_path}")


def make_line_chart(data):
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(7 * len(DATASETS), 5))
    fig.patch.set_facecolor(LIGHT_BG)
    if len(DATASETS) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, DATASETS.keys()):
        ax.set_facecolor(LIGHT_PANEL)
        scales = [2, 3, 4]
        for model in MODELS:
            psnr_vals = []
            for scale in scales:
                agg = aggregate(data, dataset, scale)
                psnr_vals.append(agg.get(model, {}).get("psnr", None))
            valid_scales = [s for s, v in zip(
                scales, psnr_vals) if v is not None]
            valid_vals = [v for v in psnr_vals if v is not None]
            if valid_vals:
                ax.plot(valid_scales, valid_vals, marker="o", linewidth=2.2,
                        markersize=7, color=COLORS[model], label=model)
                for s, v in zip(valid_scales, valid_vals):
                    ax.text(s, v + 0.05, f"{v:.2f}", ha="center", va="bottom",
                            color=COLORS[model], fontsize=8, fontweight="bold")
        ax.set_xticks([2, 3, 4])
        ax.set_xticklabels(["×2", "×3", "×4"], color=LIGHT_TEXT, fontsize=11)
        ax.tick_params(colors=LIGHT_TEXT)
        ax.set_xlabel("Facteur d'agrandissement",
                      color=LIGHT_TEXT, fontsize=10)
        ax.set_ylabel("PSNR moyen (dB)", color=LIGHT_TEXT, fontsize=10)
        ax.set_title(f"PSNR moyen — {dataset}",
                     color=LIGHT_TEXT, fontsize=12, pad=10)
        ax.spines[:].set_color(LIGHT_SPINE)
        ax.grid(color=LIGHT_GRID, linestyle="--", alpha=0.7)
        ax.legend(facecolor=LIGHT_BG, edgecolor=LIGHT_SPINE,
                  labelcolor=LIGHT_TEXT, fontsize=9)
    fig.suptitle("Évolution du PSNR selon l'échelle de super-résolution",
                 color=LIGHT_TEXT, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "psnr_by_scale.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Courbe PSNR par scale : {out_path}")


def main():
    print(f"\n{'='*60}")
    print(f"  Génération des figures de comparaison SR")
    print(f"{'='*60}\n")
    ensure_dir(FIGURES_DIR)
    data = load_csv(CSV_PATH)
    if not data:
        print("[WARN] Le CSV est vide ou absent. Lance d'abord evaluate.py.")
    print("► Génération des figures côte à côte...\n")
    for dataset_name, dataset_path in DATASETS.items():
        if not os.path.isdir(dataset_path):
            continue
        img_files = glob.glob(os.path.join(dataset_path, "*.png")) + glob.glob(
            os.path.join(dataset_path, "*.bmp")) + glob.glob(os.path.join(dataset_path, "*.jpg"))
        for img_path in sorted(img_files):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            make_comparison_figure(img_name, dataset_name, SCALE_DEMO)
    if data:
        print("\n► Génération des graphiques en barres...\n")
        make_bar_charts(data)
        print("\n► Génération du tableau récapitulatif...\n")
        make_summary_table(data)
        print("\n► Génération des courbes PSNR...\n")
        make_line_chart(data)
    print(f"\n{'='*60}")
    print(f"  Toutes les figures sont dans : {FIGURES_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
