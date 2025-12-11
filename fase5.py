# fase5_logY.py
import pickle
import torch
from tqdm import tqdm
import os

os.chdir("../../../../../../home/iogurth/gdrive/TNG300_MergerTrees")


input_pkl  = "dataset_fases/dataset_phase4_reduced_mangrove_like.pkl"
output_pkl = "dataset_fases/dataset_phase5_logY.pkl"

def load_graphs_streaming(path):
    """Iterador que lee muchos pickles consecutivos de un archivo."""
    with open(path, "rb") as f:
        while True:
            try:
                g = pickle.load(f)
                yield g
            except EOFError:
                break

def main():
    fout = open(output_pkl, "wb")

    print("Aplicando log10 a los targets...")

    count = 0
    for g in load_graphs_streaming(input_pkl):

        g.y = torch.log10(g.y)

        pickle.dump(g, fout, protocol=pickle.HIGHEST_PROTOCOL)
        count += 1
        if count % 5000 == 0:
            print(f"Procesados {count} grafos...")

    fout.close()
    print("Fase 5 completada")
    print("Guardado en:", output_pkl)

if __name__ == "__main__":
    main()
