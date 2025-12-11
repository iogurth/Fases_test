# fase7_split_datasets.py
import pickle
import random
from tqdm import tqdm

input_pkl  = "dataset_phase6_normX.pkl"

train_out  = "dataset_fases_train.pkl"
val_out    = "dataset_fases_val.pkl"
test_out   = "dataset_fases_test.pkl"

# proporciones (ajusta a gusto)
train_frac = 0.80
val_frac   = 0.10
test_frac  = 0.10

# semilla para reproducibilidad
RANDOM_SEED = 1234
random.seed(RANDOM_SEED)

def stream_load_graphs(path):
    """Generator: lee grafos uno por uno sin cargar todo en RAM."""
    with open(path, "rb") as f:
        while True:
            try:
                g = pickle.load(f)
                yield g
            except EOFError:
                break

def main():
    print("=== Fase 7: Splitting ===")

    # ----------------------------------------------------------------------
    # 1) Contar cuántos grafos hay (sin cargarlos)
    # ----------------------------------------------------------------------
    print("Contando grafos...")
    total = 0
    for _ in stream_load_graphs(input_pkl):
        total += 1

    print(f"Total de grafos = {total:,}")

    # ----------------------------------------------------------------------
    # 2) Generar lista de índices aleatoria
    # ----------------------------------------------------------------------
    indices = list(range(total))
    random.shuffle(indices)

    n_train = int(total * train_frac)
    n_val   = int(total * val_frac)
    n_test  = total - n_train - n_val

    train_idx = set(indices[:n_train])
    val_idx   = set(indices[n_train:n_train+n_val])
    test_idx  = set(indices[n_train+n_val:])

    print(f"Train: {len(train_idx):,}")
    print(f"Val:   {len(val_idx):,}")
    print(f"Test:  {len(test_idx):,}")

    # ----------------------------------------------------------------------
    # 3) Abrir archivos de salida
    # ----------------------------------------------------------------------
    fout_train = open(train_out, "wb")
    fout_val   = open(val_out, "wb")
    fout_test  = open(test_out, "wb")

    # ----------------------------------------------------------------------
    # 4) Segundo streaming: guardar cada grafo en su split
    # ----------------------------------------------------------------------
    print("Escribiendo splits...")
    idx = 0

    for g in tqdm(stream_load_graphs(input_pkl), total=total):
        if idx in train_idx:
            pickle.dump(g, fout_train, protocol=pickle.HIGHEST_PROTOCOL)
        elif idx in val_idx:
            pickle.dump(g, fout_val, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump(g, fout_test, protocol=pickle.HIGHEST_PROTOCOL)

        idx += 1

    fout_train.close()
    fout_val.close()
    fout_test.close()

    print("\n=== Fase 7 completa ===")
    print("Train →", train_out)
    print("Val   →", val_out)
    print("Test  →", test_out)


if __name__ == "__main__":
    main()
