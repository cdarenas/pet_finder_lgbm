#!/usr/bin/env python3
# Genera importancia promedio por "gain" EXCLUYENDO features RF_*.
# Uso desde C++ (main.cpp):
#   python scripts/analyze_feature_importance.py <fold_dir> <output_img>

import os, re, sys, glob
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt

# === Base features (orden exacto con el que se exportan al CLI) ===
BASE_FEATURES = [
    "Type","Age","Breed1","Breed2","Gender",
    "MaturitySize","FurLength","Health","Quantity","State",
    "Care","ColorPattern",
    "HasName","DescLength","PhotoDescCombo","FeeZero","IsBaby",
    "RescuerListingCount",
]
BASE_N = len(BASE_FEATURES)
COL_RE = re.compile(r"^Column_(\d+)$", re.IGNORECASE)

def load_true_names_if_any(fold_dir, fold_idx):
    """Si existen, usa nombres reales guardados por fold (opcional)."""
    path = os.path.join(fold_dir, f"feature_names_fold_{fold_idx}.txt")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return [ln.strip() for ln in f if ln.strip()]
    return None

def is_zero_based(lgb_names):
    return any(n == "Column_0" or n.startswith("Column_0") for n in lgb_names)

def map_and_filter(lgb_names, gains, true_names=None):
    """
    Devuelve DataFrame con columns [feature, importance_gain]
    excluyendo todo lo que sea RF_*.
    """
    rows = []
    zero_based = is_zero_based(lgb_names)
    for name, g in zip(lgb_names, gains):
        # 1) Si tengo nombres reales (archivo por fold), uso ese mapeo y filtro RF_*
        if true_names is not None:
            m = COL_RE.match(name)
            if m:
                idx = int(m.group(1))
                idx0 = idx if zero_based else idx - 1
                if 0 <= idx0 < len(true_names):
                    name = true_names[idx0]
            if str(name).startswith("RF_"):
                continue
            rows.append((str(name), float(g)))
            continue

        # 2) Sin nombres reales: mapeo por índice y ME QUEDO SOLO con las primeras BASE_N
        m = COL_RE.match(name)
        if not m:
            # Si no es Column_i (raro), sólo lo dejo si no parece RF_*
            if not str(name).startswith("RF_"):
                rows.append((str(name), float(g)))
            continue

        idx = int(m.group(1))
        idx0 = idx if zero_based else idx - 1
        if 0 <= idx0 < BASE_N:
            rows.append((BASE_FEATURES[idx0], float(g)))
        # Si idx0 >= BASE_N => es RF_* u otra derivada al final → excluir

    return pd.DataFrame(rows, columns=["feature", "importance_gain"])

def main():
    if len(sys.argv) < 3:
        print("Uso: python analyze_feature_importance.py <fold_dir> <output_img>")
        sys.exit(1)

    fold_dir = sys.argv[1]
    out_img  = sys.argv[2]

    # Detectar folds disponibles (model_fold_*.txt)
    model_paths = sorted(glob.glob(os.path.join(fold_dir, "model_fold_*.txt")))
    if not model_paths:
        print(f"[ERROR] No se encontraron modelos en {fold_dir}")
        sys.exit(1)

    dfs = []
    for mp in model_paths:
        # fold index
        m = re.search(r"model_fold_(\d+)\.txt$", mp)
        fold_idx = int(m.group(1)) if m else -1

        booster = lgb.Booster(model_file=mp)
        gains = booster.feature_importance(importance_type="gain")
        names = booster.feature_name()

        true_names = load_true_names_if_any(fold_dir, fold_idx)
        df = map_and_filter(names, gains, true_names=true_names)
        if df.empty:
            # nada tras filtrar RF; seguir
            continue
        df["fold"] = fold_idx
        dfs.append(df)

    if not dfs:
        print("[WARN] No hay importancias (tras filtrar RF_*). Genero gráfico vacío.")
        plt.figure(figsize=(8, 4))
        plt.title("Importancia (sin RF_*)")
        plt.tight_layout()
        plt.savefig(out_img, dpi=180)
        sys.exit(0)

    imp = pd.concat(dfs, ignore_index=True)
    mean_imp = (imp.groupby("feature", as_index=False)["importance_gain"]
                  .mean()
                  .sort_values("importance_gain", ascending=False))

    # Grafico top-30 para que sea legible
    top = mean_imp.head(30)

    plt.figure(figsize=(12, max(4, 0.4 * len(top) + 2)))
    plt.barh(top["feature"], top["importance_gain"])
    plt.gca().invert_yaxis()
    plt.title("Importancia promedio por gain (excluye RF_*)")
    plt.xlabel("gain promedio")
    plt.tight_layout()
    plt.savefig(out_img, dpi=180)
    print(f"[OK] Guardado: {out_img}")

if __name__ == "__main__":
    main()
