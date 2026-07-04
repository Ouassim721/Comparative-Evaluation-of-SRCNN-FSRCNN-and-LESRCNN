"""
=============================================================
  Script de visualisation comparative — Super-Résolution CNN
  Génère :
    1. Figures comparatives côte à côte (LR / SRCNN / FSRCNN / LESRCNN / HR)
    2. Graphiques PSNR et SSIM par modèle et dataset
    3. Tableau récapitulatif en image
=============================================================

UTILISATION (après avoir lancé evaluate.py) :
    python visualize.py

PARAMÈTRES MODIFIABLES ci-dessous dans la section CONFIGURATION.
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

matplotlib.use("Agg")   # pas besoin d'écran

# ===========================================================
#  CONFIGURATION
# ===========================================================

DATASETS = {
    "Set5":  "datasets/Set5",
    "Set14": "datasets/Set14",
}

RESULTS_DIR = "results"          # dossier généré par evaluate.py
FIGURES_DIR = "figures"          # dossier de sortie des figures
CSV_PATH = "results/resultats.csv"

SCALE_DEMO = 2                  # échelle utilisée pour les figures côte à côte
MODELS = ["Bicubic", "SRCNN", "FSRCNN", "LESRCNN"]
COLORS = {
    "Bicubic": "#7F8C8D",
    "SRCNN":   "#C0392B",
    "FSRCNN":  "#2471A3",
    "LESRCNN": "#1E8449",
}

# ---- Palette "mode clair" pour figures d'article ----
LIGHT_BG = "#FFFFFF"        # fond de figure
LIGHT_PANEL = "#F4F4F4"     # fond des axes / panneaux
LIGHT_TEXT = "#1A1A1A"      # texte principal (quasi noir)
LIGHT_GRID = "#CCCCCC"      # grille discrète
LIGHT_SPINE = "#888888"     # bordures des axes
# couleur "Référence" (remplace le jaune illisible sur blanc)
REF_COLOR = "#8E6E00"

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


def load_csv(path):
    """Charge le CSV de résultats en dictionnaire."""
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
    """Calcule la moyenne PSNR/SSIM par modèle pour un dataset et une échelle."""
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
    """
    Génère une figure côte à côte :
    [LR Bicubic] [SRCNN] [FSRCNN] [LESRCNN] [HR Original]
    Avec le PSNR affiché sous chaque image.
    """
    # Chercher l'image HR
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

    # Bicubique
    img_bic = make_lr_bicubic(img_hr, scale)

    # Images SR depuis results/
    sr_images = {}
    for model in ["SRCNN", "FSRCNN", "LESRCNN"]:
        sr_path = os.path.join(
            RESULTS_DIR, model, f"{dataset_name}_{img_name}_x{scale}.png")
        if os.path.isfile(sr_path):
            sr_images[model] = cv2.imread(sr_path, cv2.IMREAD_GRAYSCALE)
        else:
            sr_images[model] = None

    # Convertir HR en canal Y pour affichage uniforme
    hr_y = bgr2ycbcr_y(img_hr)
    bic_y = bgr2ycbcr_y(img_bic)

    # Construire la figure
    panels = [
        ("LR (Bicubic)", bic_y),
        ("SRCNN",        sr_images.get("SRCNN")),
        ("FSRCNN",       sr_images.get("FSRCNN")),
        ("LESRCNN",      sr_images.get("LESRCNN")),
        ("HR Original",  hr_y),
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
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            if title not in ("HR Original", "LR (Bicubic)"):
                p = calc_psnr(hr_y, img[:hr_y.shape[0],
                              :hr_y.shape[1]], data_range=255)
                s = calc_ssim(hr_y, img[:hr_y.shape[0],
                              :hr_y.shape[1]], data_range=255)
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

    fig.suptitle(
        f"Comparaison SR — {img_name} (Scale ×{scale})",
        color=LIGHT_TEXT, fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    out_path = os.path.join(
        FIGURES_DIR, f"comparison_{dataset_name}_{img_name}_x{scale}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Figure sauvegardée : {out_path}")


# ===========================================================
#  FIGURE 2 — GRAPHIQUES PSNR / SSIM PAR SCALE
# ===========================================================

def make_bar_charts(data):
    """
    Génère des graphiques en barres groupées PSNR et SSIM
    pour chaque dataset et chaque scale.
    """
    scales = [2, 3, 4]
    dataset_names = list(DATASETS.keys())

    for dataset in dataset_names:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(LIGHT_BG)
        fig.suptitle(f"PSNR & SSIM moyen — {dataset}", color=LIGHT_TEXT,
                     fontsize=14, fontweight="bold", y=1.01)

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

                bars = ax.bar(
                    x + offsets[i], values, bar_w,
                    label=model, color=COLORS[model],
                    alpha=0.92, edgecolor=LIGHT_TEXT, linewidth=0.4
                )
                # Annoter les valeurs
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (0.05 if metric == "psnr" else 0.001),
                            f"{val:.2f}" if metric == "psnr" else f"{val:.3f}",
                            ha="center", va="bottom",
                            color=LIGHT_TEXT, fontsize=7, fontweight="bold"
                        )

            ax.set_xticks(x)
            ax.set_xticklabels([f"×{s}" for s in scales],
                               color=LIGHT_TEXT, fontsize=11)
            ax.set_ylabel("PSNR (dB)" if metric == "psnr" else "SSIM",
                          color=LIGHT_TEXT, fontsize=11)
            ax.set_title("PSNR moyen" if metric == "psnr" else "SSIM moyen",
                         color=LIGHT_TEXT, fontsize=12, pad=10)
            ax.tick_params(colors=LIGHT_TEXT)
            ax.spines[:].set_color(LIGHT_SPINE)
            ax.set_ylim(bottom=0)
            ax.legend(
                facecolor=LIGHT_BG, edgecolor=LIGHT_SPINE,
                labelcolor=LIGHT_TEXT, fontsize=9
            )
            ax.grid(axis="y", color=LIGHT_GRID, linestyle="--", alpha=0.7)

        plt.tight_layout()
        out_path = os.path.join(FIGURES_DIR, f"barchart_{dataset}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  ✓ Graphique barres : {out_path}")


# ===========================================================
#  FIGURE 3 — TABLEAU RÉCAPITULATIF VISUEL
# ===========================================================

def make_summary_table(data):
    """
    Génère un tableau visuel synthétique de tous les résultats
    (1 ligne par modèle × scale × dataset).
    """
    rows = []
    for dataset in DATASETS.keys():
        for scale in [2, 3, 4]:
            agg = aggregate(data, dataset, scale)
            for model in MODELS:
                if model in agg:
                    rows.append([
                        dataset, f"×{scale}", model,
                        f"{agg[model]['psnr']:.4f}",
                        f"{agg[model]['ssim']:.4f}",
                    ])

    if not rows:
        print("  [SKIP] Aucune donnée pour le tableau récapitulatif.")
        return

    headers = ["Dataset", "Scale", "Modèle", "PSNR moy (dB)", "SSIM moy"]
    n_rows = len(rows) + 1  # +1 header
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

    # En-têtes
    for j, (header, x) in enumerate(zip(headers, x_positions)):
        ax.text(
            x / total_w + col_widths[j] / total_w / 2,
            1 - row_h / 2,
            header, ha="center", va="center",
            transform=ax.transAxes,
            color=LIGHT_TEXT, fontsize=10, fontweight="bold"
        )

    # Lignes
    for i, row in enumerate(rows):
        y = 1 - row_h * (i + 1.5)
        # Couleur alternée (bandes discrètes en mode clair)
        bg_color = LIGHT_PANEL if i % 2 == 0 else LIGHT_BG
        # Surligner le meilleur PSNR par groupe (dataset+scale)
        model_color = COLORS.get(row[2], LIGHT_TEXT)

        rect = FancyBboxPatch(
            (0, y - row_h * 0.45), 1, row_h * 0.9,
            boxstyle="round,pad=0.005",
            facecolor=bg_color, edgecolor="none",
            transform=ax.transAxes, zorder=0
        )
        ax.add_patch(rect)

        for j, (cell, x) in enumerate(zip(row, x_positions)):
            color = model_color if j == 2 else LIGHT_TEXT
            fw = "bold" if j in (2, 3) else "normal"
            ax.text(
                x / total_w + col_widths[j] / total_w / 2,
                y,
                cell, ha="center", va="center",
                transform=ax.transAxes,
                color=color, fontsize=9, fontweight=fw
            )

    ax.set_title(
        "Résultats comparatifs — Super-Résolution CNN",
        color=LIGHT_TEXT, fontsize=13, fontweight="bold", pad=12
    )

    out_path = os.path.join(FIGURES_DIR, "summary_table.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Tableau récapitulatif : {out_path}")


# ===========================================================
#  FIGURE 4 — COURBE PSNR SELON LE SCALE (LINE CHART)
# ===========================================================

def make_line_chart(data):
    """Courbe PSNR moyen (toutes images confondues) par scale et par modèle."""
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

            # Filtrer les None
            valid_scales = [s for s, v in zip(
                scales, psnr_vals) if v is not None]
            valid_vals = [v for v in psnr_vals if v is not None]

            if valid_vals:
                ax.plot(valid_scales, valid_vals,
                        marker="o", linewidth=2.2, markersize=7,
                        color=COLORS[model], label=model)
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


# ===========================================================
#  PROGRAMME PRINCIPAL
# ===========================================================

def main():
    print(f"\n{'='*60}")
    print(f"  Génération des figures de comparaison SR")
    print(f"{'='*60}\n")

    ensure_dir(FIGURES_DIR)

    # Charger les résultats CSV
    data = load_csv(CSV_PATH)
    if not data:
        print("[WARN] Le CSV est vide ou absent. Lance d'abord evaluate.py.")
        print("       Les figures de comparaison côte à côte seront générées")
        print("       si les images HR et SR existent.\n")

    # ── Figure 1 : Comparaisons côte à côte ──
    print("► Génération des figures côte à côte...\n")
    for dataset_name, dataset_path in DATASETS.items():
        if not os.path.isdir(dataset_path):
            continue
        img_files = (
            glob.glob(os.path.join(dataset_path, "*.png")) +
            glob.glob(os.path.join(dataset_path, "*.bmp")) +
            glob.glob(os.path.join(dataset_path, "*.jpg"))
        )
        for img_path in sorted(img_files):
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            make_comparison_figure(img_name, dataset_name, SCALE_DEMO)

    # ── Figure 2 : Graphiques en barres ──
    if data:
        print("\n► Génération des graphiques en barres...\n")
        make_bar_charts(data)

        # ── Figure 3 : Tableau récapitulatif ──
        print("\n► Génération du tableau récapitulatif...\n")
        make_summary_table(data)

        # ── Figure 4 : Courbes PSNR par scale ──
        print("\n► Génération des courbes PSNR...\n")
        make_line_chart(data)

    print(f"\n{'='*60}")
    print(f"  Toutes les figures sont dans : {FIGURES_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
