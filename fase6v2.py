import pickle
import torch
import numpy as np
import os

os.chdir("../TNG_SUBLINK_300-1")

input_pkl  = "../DATASET/dataset_phase5_logY.pkl"
output_pkl = "../DATASET/dataset_phase6_normX.pkl"
stats_out  = "../DATASET/phase6_norm_stats.npz"

SNAP_COL = 11   # <- Reemplaza por el índice que te dio el código de verificación

# 1. PEGAR AQUÍ EL DICCIONARIO COMPLETO (resumido por espacio, ponlos todos)
snap_to_redshift = {
    0: 20.05, 1: 14.99, 2: 11.98, 3: 10.98, 4: 10.00, 5: 9.39, 6: 9.00, 7: 8.45, 8: 8.01, 9: 7.60,
    10: 7.24, 11: 7.01, 12: 6.49, 13: 6.01, 14: 5.85, 15: 5.53, 16: 5.23, 17: 5.00, 18: 4.66, 19: 4.43,
    20: 4.18, 21: 4.01, 22: 3.71, 23: 3.49, 24: 3.28, 25: 3.01, 26: 2.90, 27: 2.73, 28: 2.58, 29: 2.44,
    30: 2.32, 31: 2.21, 32: 2.10, 33: 2.00, 34: 1.90, 35: 1.82, 36: 1.74, 37: 1.67, 38: 1.60, 39: 1.53,
    40: 1.50, 41: 1.41, 42: 1.36, 43: 1.30, 44: 1.25, 45: 1.21, 46: 1.15, 47: 1.11, 48: 1.07, 49: 1.04,
    50: 1.00, 51: 0.95, 52: 0.92, 53: 0.89, 54: 0.85, 55: 0.82, 56: 0.79, 57: 0.76, 58: 0.73, 59: 0.70,
    60: 0.68, 61: 0.64, 62: 0.62, 63: 0.60, 64: 0.58, 65: 0.55, 66: 0.52, 67: 0.50, 68: 0.48, 69: 0.46,
    70: 0.44, 71: 0.42, 72: 0.40, 73: 0.38, 74: 0.36, 75: 0.35, 76: 0.33, 77: 0.31, 78: 0.30, 79: 0.27,
    80: 0.26, 81: 0.24, 82: 0.23, 83: 0.21, 84: 0.20, 85: 0.18, 86: 0.17, 87: 0.15, 88: 0.14, 89: 0.13,
    90: 0.11, 91: 0.10, 92: 0.08, 93: 0.07, 94: 0.06, 95: 0.05, 96: 0.03, 97: 0.02, 98: 0.01, 99: 0.00,
}

# 2. CREAR TENSOR DE MAPEO PARA PYTORCH
max_snap = max(snap_to_redshift.keys())
snap_map_tensor = torch.zeros(max_snap + 1, dtype=torch.float32)
for k, v in snap_to_redshift.items():
    snap_map_tensor[k] = v

def map_snap_to_redshift(g):
    """Función para convertir la columna SnapNum a Redshift vectorizadamente"""
    # Extraer la columna como enteros largos para usar de índices
    snaps = g.x[:, SNAP_COL].long()
    # Limitar por seguridad para evitar IndexError
    snaps = torch.clamp(snaps, 0, max_snap)
    # Reemplazar la columna con los valores de redshift
    g.x[:, SNAP_COL] = snap_map_tensor[snaps]
    return g

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
    """Primer pase: acumula medias y std de X (excepto Redshift)."""
    print("→ Pase 1: calculando medias y std...")

    count = 0
    sum_x = None
    sum_x2 = None

    for g in stream_graphs(path):
        # APLICAR MAPEO ANTES DE CALCULAR ESTADÍSTICAS
        g = map_snap_to_redshift(g)
        
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

    std = torch.clamp(std, min=1e-6)

    # NO NORMALIZAR REDSHIFT (Lo mantenemos en su escala física 0.0 - 20.05)
    std[SNAP_COL] = 1.0
    mu[SNAP_COL]  = 0.0

    np.savez(stats_out, mu=mu.numpy(), std=std.numpy())
    print("Stats guardadas en", stats_out)

    return mu.float(), std.float()


def pass2_apply_norm(input_path, output_path, mu, std):
    """Segundo pase: normaliza X y reescribe el dataset."""
    print("Pase 2: escribiendo dataset normalizado...")

    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        while True:
            try:
                g = pickle.load(fin)

                # APLICAR MAPEO ANTES DE NORMALIZAR
                g = map_snap_to_redshift(g)

                # aplicar normalización
                g.x = (g.x - mu) / std
                
                pickle.dump(g, fout, protocol=pickle.HIGHEST_PROTOCOL)
            except EOFError:
                break
    print("✓ Dataset normalizado en", output_path)

if __name__ == "__main__":
    mu, std = pass1_compute_stats(input_pkl)
    pass2_apply_norm(input_pkl, output_pkl, mu, std)
