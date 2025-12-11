# fase6_normalize_X_ignore_snapnum.py
import pickle
import torch
import numpy as np
import os

os.chdir("../../../../../../home/iogurth/gdrive/TNG300_MergerTrees")

input_pkl  = "dataset_fases/dataset_phase5_logY.pkl"
output_pkl = "dataset_fases/dataset_phase6_normX.pkl"
stats_out  = "dataset_fases/phase6_norm_stats.npz"

SNAP_COL = 13   # <- tu columna SnapNum

def stream_graphs(path):
    """Generador que lee grafos uno por uno desde un PKL grande."""
    with open(path, "rb") as f:
        while True:
            try:
                g = pickle.load(f)
                yield g
            except EOFError:
                break

def pass1_compute_stats(path):
    """Primer pase: acumula medias y std de X (excepto SnapNum)."""
    print("→ Pase 1: calculando medias y std...")

    count = 0
    sum_x = None
    sum_x2 = None

    for g in stream_graphs(path):
        x = g.x
        if sum_x is None:
            D = x.shape[1]
            sum_x  = torch.zeros(D, dtype=torch.float64)
            sum_x2 = torch.zeros(D, dtype=torch.float64)

        sum_x  += x.sum(dim=0).double()
        sum_x2 += (x.double() ** 2).sum(dim=0)
        count  += x.shape[0]

    mu  = sum_x / count
    var = (sum_x2 / count) - mu**2
    std = torch.sqrt(torch.clamp(var, min=1e-12))

    # Forzar std mínimo
    std = torch.clamp(std, min=1e-6)

    # NO NORMALIZAR SnapNum
    std[SNAP_COL] = 1.0
    mu[SNAP_COL]  = 0.0

    np.savez(stats_out, mu=mu.numpy(), std=std.numpy())
    print("✓ Stats guardadas en", stats_out)

    return mu.float(), std.float()


def pass2_apply_norm(input_path, output_path, mu, std):
    """Segundo pase: normaliza X y reescribe el dataset."""
    print("→ Pase 2: escribiendo dataset normalizado...")

    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        while True:
            try:
                g = pickle.load(fin)

                # aplicar normalización
                g.x = (g.x - mu) / std

                # snapnum NO se normalizó, así que está intacto
                pickle.dump(g, fout, protocol=pickle.HIGHEST_PROTOCOL)

            except EOFError:
                break

    print("✓ Fase 6 completada.")
    print("→ Output:", output_path)


def main():
    mu, std = pass1_compute_stats(input_pkl)
    pass2_apply_norm(input_pkl, output_pkl, mu, std)

if __name__ == "__main__":
    main()
