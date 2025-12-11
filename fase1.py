import h5py
import numpy as np
from tqdm import tqdm
import os


os.chdir("../../../../../../home/iogurth/gdrive/TNG300_MergerTrees")


# Archivos de entrada (puedes ajustarlo)
files = [f"tree_extended.{i}.hdf5" for i in range(125)]
output_file = "dataset_fases/phase1_structural.h5"
index_file = "dataset_fases/phase1_state.txt"

# Campos estructurales que definen la jerarquía
structure_fields = [
    "SubhaloID", "TreeID", "DescendantID",
    "FirstProgenitorID", "NextProgenitorID",
    "SnapNum", "GroupFirstSub"
]

# -------------------------------
# Reanudación si se interrumpe
# -------------------------------
if os.path.exists(index_file):
    with open(index_file) as f:
        processed = set(map(str.strip, f.readlines()))
else:
    processed = set()

# -------------------------------
# Escritura incremental en HDF5
# -------------------------------
with h5py.File(output_file, "a") as out:
    for file in tqdm(files, desc="Procesando archivos estructurales"):
        if file in processed:
            continue
        with h5py.File(file, "r") as f:
            grp = out.create_group(file)
            for field in structure_fields:
                if field in f.keys():
                    data = f[field][:]
                    grp.create_dataset(field, data=data, compression="gzip", compression_opts=4)
        with open(index_file, "a") as fstate:
            fstate.write(file + "\n")
print("Fase 1 completada: estructura guardada en", output_file)
