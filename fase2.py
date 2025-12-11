import h5py
import numpy as np
import pickle
from tqdm import tqdm
import os

os.chdir("../../../../../../home/iogurth/gdrive/TNG300_MergerTrees")

# -------------------------------
# CONFIGURACIÓN
# -------------------------------
phase1_file = "dataset_fases/phase1_structural.h5"       # archivo existente
output_file = "dataset_fases/phase2_few_features.h5"    # archivo nuevo
state_file = "dataset_fases/phase2_state.pkl"            # estado de avance
tree_files = [f"tree_extended.{i}.hdf5" for i in range(125)]  # rango total de archivos

# Campos físicos a agregar
feature_fields = [
    #"Mass", "SubhaloMassType", "SubhaloVmax", "SubhaloVelDisp", "SubhaloSpin",
    #"SubhaloHalfmassRad", "SubhaloPos", "SubhaloVel",
    #"Group_M_Crit200", "Group_M_Crit500", "Group_M_Mean200",
    #"SubhaloMass", "SnapNum", "SubhaloID"
    #"SubhaloWindMass", "SubhaloStarMetallicity", "SubhaloBHMass", "SubhaloGasMetalFractions"
    #"GroupBHMass", "GroupBHMdot", "GroupVel", "GroupWindMass"


    #"SubhaloWindMass", "SubhaloStarMetallicity", "SubhaloBHMass", "SubhaloGasMetalFractions"
    #"GroupBHMass", "GroupBHMdot", "GroupWindMass"


    #"SubhaloMassType", "SnapNum", "FirstProgenitorID", "NextProgenitorID", "Group_M_Crit200", 
    #"Group_R_Crit200", "SubhaloHalfmassRad", "SubhaloVelDisp",
    #"SubhaloID", "SubhaloVmax", "SubhaloSpin", "Group_M_Crit500", "Group_M_Mean200",
    #"GroupPos", "SubhaloVel", "GroupVel", "Mass", "SubhaloPos"

    "SubhaloBHMass", "SubhaloGasMetallicity", "SubhaloSFR", "SubhaloSFRinRad"
]

# -------------------------------
# CARGAR ESTADO PREVIO
# -------------------------------
done_files = set()
if os.path.exists(state_file):
    with open(state_file, "rb") as f:
        done_files = pickle.load(f)
    print(f"Reanudando desde estado previo ({len(done_files)} archivos completados)")

# -------------------------------
# ABRIR ARCHIVO ESTRUCTURAL (lectura y escritura)
# -------------------------------
with h5py.File(phase1_file, "r") as f_in, h5py.File(output_file, "a") as f_out:
    
    # Copiar datasets de la estructura si es la primera vez
    #if "TreeID" not in f_out:
    #    for name in f_in.keys():
    #        f_in.copy(name, f_out)
    #    print("Copiada estructura base al nuevo archivo.")

    # Crear grupos para features si no existen
    if "Features" not in f_out:
        f_out.create_group("Features")

    # -------------------------------
    # PROCESAR CADA ARCHIVO HDF5 ORIGINAL
    # -------------------------------
    for file in tqdm(tree_files, desc="Agregando features físicos"):
        if file in done_files:
            continue
        if not os.path.exists(file):
            print(f"Archivo no encontrado: {file}")
            continue

        with h5py.File(file, "r") as f_tree:
            # Leer SubhaloID
            sub_ids = f_tree["SubhaloID"][:]
            n = len(sub_ids)
            # Crear diccionario de índices
            subid_to_idx = {sid: i for i, sid in enumerate(sub_ids)}

            # Crear arrays temporales para features
            for feat in feature_fields:
                data = f_tree[feat][:]
                # Guardar en un dataset en el archivo final
                if feat not in f_out["Features"]:
                    maxshape = (0,) + data.shape[1:] if data.ndim > 1 else (0,)
                    f_out["Features"].create_dataset(
                        feat, shape=maxshape, maxshape=(None,)+maxshape[1:], dtype=data.dtype, chunks=True
                    )
                dset = f_out["Features"][feat]
                dset.resize((dset.shape[0] + n,) + dset.shape[1:])
                dset[-n:] = data

        # Marcar como completado
        done_files.add(file)
        with open(state_file, "wb") as f:
            pickle.dump(done_files, f)

print("Fase 2 completada.")
print(f"Archivo guardado: {output_file}")
