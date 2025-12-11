# fase3_fixed_all_targets_thresholds.py
import h5py
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import pickle
import os
import re
import json

# ====== Ajusta rutas según tu entorno ======
os.chdir("../../../../../../home/iogurth/gdrive/TNG300_MergerTrees")

input_file = "dataset_fases/phase2_few_features.h5"
output_file = "dataset_fases/dataset_phase3_mangrove_like_all_targets_FULLRAW_FIX_extra_info.pkl"
state_file = "dataset_fases/state_phase3_thresholds.json"

# ====== Hiperparámetros (EDITA AQUÍ) ======
# Unidad: SubhaloMassType[:,4] en tu phase2 está en [1e10 Msun/h]
mass_cut = 0.0        # umbral para la raíz (smass) usado en selección de centrals
mass_node_cut = 0.0  # umbral mínimo para nodos individuales (smass)
min_nodes = 15         # guardar solo árboles con >= min_nodes

# Thresholds por target (las claves deben corresponder al orden de 'y' que usas)
# En este script la y = [smass, mcold, mbh, zcold, sfrinst, sfravg]
target_thresholds = {
    "smass": 0.0,   # por defecto igual a mass_cut
    "mcold": 0.0,
    "mbh": 0.0,
    "zcold": 0.0,
    "sfrinst": 0.0,
    "sfravg": 0.0
}

# general_target_threshold = None  -> usa target_thresholds individuales
general_target_threshold = 0.0  # ejemplo: 0.0

# Si require_any=True guarda el árbol si al menos 1 target supera su umbral.
# Si require_any=False guarda solo si TODOS los targets superan su umbral.
require_any = False

test_mode = False
test_n_groups = 3

log_every_graphs = 5000

# ====== Helpers ======
def extract_number(name):
    m = re.search(r"tree_extended\.(\d+)\.hdf5", name)
    return int(m.group(1)) if m else -1

def safe_get_feature(arr_group, key, start, end, fallback_shape=None, dtype=np.float32):
    """Extrae arr_group[key][start:end] si existe; si no, devuelve ceros shape apropiado."""
    if key in arr_group:
        return arr_group[key][start:end]
    else:
        length = end - start
        if fallback_shape is None:
            return np.zeros((length,), dtype=dtype)
        else:
            return np.zeros((length,)+tuple(fallback_shape), dtype=dtype)

def passes_target_thresholds(root_vals_dict, target_thresholds_local, general_threshold=None, require_any=True):
    """
    root_vals_dict: {'smass': val, 'mcold': val, ...}
    target_thresholds_local: dict de umbrales por target
    general_threshold: si no None, se usa para todos
    return: True/False
    """
    if general_threshold is not None:
        # aplicar mismo umbral a todos
        checks = [ (val > general_threshold) for val in root_vals_dict.values() ]
    else:
        checks = []
        for k, val in root_vals_dict.items():
            thr = target_thresholds_local.get(k, 0.0)
            checks.append( (val > thr) )
    if require_any:
        return any(checks)
    else:
        return all(checks)

# ====== Construcción de árbol ======
def build_tree(root_sid, subid_to_idx, FirstProg_local, NextProg_local, Desc_local,
               smass_local, mcold_local, mbh_local, zcold_local, sfrinst_local, sfravg_local,
               SnapNum_local, feature_slices, mass_node_cut, min_nodes):

    stack = [root_sid]
    visited = set()
    nodes = []       # (sid, features, local_idx)
    edges = []

    while stack:
        sid = stack.pop()
        if sid in visited:
            continue
        if sid not in subid_to_idx:
            continue
        visited.add(sid)
        local_idx = subid_to_idx[sid]

        node_smass = float(smass_local[local_idx])
        if np.isnan(node_smass) or node_smass < mass_node_cut:
            continue

        feats_parts = []
        for arr, key in feature_slices:
            val = arr[local_idx]
            feats_parts.append(np.atleast_1d(val).ravel())

        flat_feats = np.concatenate(feats_parts, dtype=np.float32)
        nodes.append((sid, flat_feats, local_idx))

        # main progenitor
        prog_sid = int(FirstProg_local[local_idx])
        if prog_sid >= 0 and prog_sid in subid_to_idx:
            edges.append((prog_sid, sid))
            stack.append(prog_sid)

        # secondary progenitors
        nxt_sid = int(NextProg_local[local_idx])
        while nxt_sid >= 0:
            if nxt_sid in subid_to_idx:
                edges.append((nxt_sid, sid))
                stack.append(nxt_sid)
                nxt_local = subid_to_idx[nxt_sid]
                nxt_sid = int(NextProg_local[nxt_local])
            else:
                break

    if len(nodes) < min_nodes:
        return None

    # construir features
    feat_matrix = np.array([f for (_, f, _) in nodes], dtype=np.float32)
    x = torch.tensor(feat_matrix, dtype=torch.float32)

    # mapping original SubhaloID -> fila del grafo
    idmap = {sid: i for i, (sid, _, _) in enumerate(nodes)}

    edges_filtered = [(idmap[u], idmap[v]) for (u, v) in edges if u in idmap and v in idmap]
    if len(edges_filtered) == 0:
        return None
    edge_index = torch.tensor(edges_filtered, dtype=torch.long).t().contiguous()

    # --- TARGET ---
    root_local_idx = subid_to_idx[root_sid]
    y = torch.tensor([
        float(smass_local[root_local_idx]),
        float(mcold_local[root_local_idx]),
        float(mbh_local[root_local_idx]),
        float(zcold_local[root_local_idx]),
        float(sfrinst_local[root_local_idx]),
        float(sfravg_local[root_local_idx])
    ], dtype=torch.float32)

    # =====================
    #  NUEVOS ATRIBUTOS
    # =====================

    # 1. SubhaloID alineados
    subhalo_ids = torch.tensor([sid for sid, _, _ in nodes], dtype=torch.long)

    # 2. SnapNum alineado al grafo
    snapnums = torch.tensor(
        [SnapNum_local[local_idx] for (_, _, local_idx) in nodes],
        dtype=torch.int16
    )

    # 3. FirstProgenitorID → mapa al grafo
    first_prog = torch.tensor([
        idmap.get(int(FirstProg_local[local_idx]), -1)
        for (_, _, local_idx) in nodes
    ], dtype=torch.long)

    # 4. NextProgenitorID → mapa al grafo
    next_prog = torch.tensor([
        idmap.get(int(NextProg_local[local_idx]), -1)
        for (_, _, local_idx) in nodes
    ], dtype=torch.long)

    # 5. DescendantID → mapa (si existe)
    if Desc_local is not None:
        descendant = torch.tensor([
            idmap.get(int(Desc_local[local_idx]), -1)
            for (_, _, local_idx) in nodes
        ], dtype=torch.long)
    else:
        descendant = None

    # construir Data
    data = Data(x=x, edge_index=edge_index, y=y)

    # añadir atributos extra
    data.subhalo_ids = subhalo_ids
    data.snapnum = snapnums
    data.first_progenitor = first_prog
    data.next_progenitor = next_prog
    if descendant is not None:
        data.descendant = descendant

    return data


# ====== MAIN ======
def main():
    if not os.path.exists("dataset_fases"):
        os.makedirs("dataset_fases")

    with h5py.File(input_file, "r") as f:
        # enumerar y ordenar grupos
        groups = [k for k in f.keys() if k.startswith("tree_extended.")]
        groups = sorted(groups, key=extract_number)
        if test_mode:
            groups = groups[:test_n_groups]
        print(f"Total de grupos a procesar: {len(groups)}")

        # features disponibles
        feature_keys = list(f.get("Features", {}).keys())
        print("Features disponibles (Features/):", feature_keys)

        # offsets globales por grupo (para el bloque Features)
        offsets = {}
        cur = 0
        for g in groups:
            n = int(f[g]["SubhaloID"].shape[0])
            offsets[g] = (cur, cur + n)
            cur += n

        # reanudación
        done_idx = 0
        count_graphs = 0
        if os.path.exists(state_file):
            with open(state_file, "r") as sf:
                st = json.load(sf)
                done_idx = int(st.get("done_idx", 0))
                count_graphs = int(st.get("count", 0))
            print(f"Reanudando desde grupo #{done_idx}, grafos previos: {count_graphs}")

        # keys para features por nodo (ajusta según quieras)
        node_feature_keys = [
            "Concentration", #,)
            "GroupPos", #, 3)
            "GroupVel", #, 3)
            "Group_M_Crit200", #,)
            "Group_M_Crit500", #,)
            "Group_M_Mean200", #,)
            "Group_R_Crit200", #,)
            "Mass", #,)
            "Rs", #,)
            "SnapNum", 
            "SpinBullock",
            "SubhaloHalfmass",
            #"SubhaloID",
            "SubhaloMass",
            "SubhaloPos",
            "SubhaloSpin",
            "SubhaloVel",
            "SubhaloVelDisp"
        ]

        with open(output_file, "ab") as fout:
            for gi in range(done_idx, len(groups)):
                gname = groups[gi]
                grp = f[gname]
                start, end = offsets[gname]
                n_local = int(end - start)

                # preparar feature_slices (solo lecturas para este grupo)
                feature_slices = []
                for key in node_feature_keys:
                    if key == "SubhaloMassType" and "SubhaloMassType" in f["Features"]:
                        arr = f["Features"]["SubhaloMassType"][start:end]  # (n_local,6)
                        feature_slices.append((arr, key))
                    else:
                        arr = safe_get_feature(f["Features"], key, start, end)
                        feature_slices.append((arr, key))

                # SubhaloMassType obligatorio para sacar smass y mcold (si no está, safe_get dará zeros)
                SubhaloMassType_slice = safe_get_feature(f["Features"], "SubhaloMassType", start, end, fallback_shape=(6,))
                smass_slice = SubhaloMassType_slice[:, 4]   # masa estelar
                mcold_slice = SubhaloMassType_slice[:, 0]   # gas (aprox)

                # otras quantities (fallback si no existen)
                mbh_slice = safe_get_feature(f["Features"], "SubhaloBHMass", start, end)
                zcold_slice = safe_get_feature(f["Features"], "SubhaloGasMetallicity", start, end)
                sfrinst_slice = safe_get_feature(f["Features"], "SubhaloSFR", start, end)
                sfravg_slice = safe_get_feature(f["Features"], "SubhaloSFRinRad", start, end)

                # arrays topológicos locales
                TreeID = grp["TreeID"][:]
                SubhaloID = grp["SubhaloID"][:]
                SnapNum_local = grp["SnapNum"][:]
                FirstProg_local = grp["FirstProgenitorID"][:]
                NextProg_local = grp["NextProgenitorID"][:]
                Descendant_local = grp["DescendantID"][:]

                # mapping SubhaloID -> índice local
                subid_to_idx = {int(sid): i for i, sid in enumerate(SubhaloID)}

                # seleccionar raíz más reciente por TreeID
                unique_tids = np.unique(TreeID)
                centrals_idx = []
                for tid in unique_tids:
                    mask = TreeID == tid
                    local_indices = np.where(mask)[0]
                    latest_local = local_indices[np.argmax(SnapNum_local[local_indices])]
                    if smass_slice[latest_local] >= mass_cut:
                        centrals_idx.append(int(latest_local))

                print(f"{gname}: {len(centrals_idx)} centrals detectados (start={start:,})")

                # construir y guardar grafos uno a uno
                saved_in_group = 0
                for local_idx in centrals_idx:
                    root_sid = int(SubhaloID[local_idx])

                    # build root target values for threshold check
                    root_vals = {
                        "smass": float(smass_slice[local_idx]),
                        "mcold": float(mcold_slice[local_idx]),
                        "mbh": float(mbh_slice[local_idx]) if mbh_slice is not None else 0.0,
                        "zcold": float(zcold_slice[local_idx]) if zcold_slice is not None else 0.0,
                        "sfrinst": float(sfrinst_slice[local_idx]) if sfrinst_slice is not None else 0.0,
                        "sfravg": float(sfravg_slice[local_idx]) if sfravg_slice is not None else 0.0
                    }

                    # decide si guarda (según thresholds)
                    if not passes_target_thresholds(root_vals, target_thresholds, general_threshold=general_target_threshold, require_any=require_any):
                        continue

                    g = build_tree(root_sid, subid_to_idx,
                                   FirstProg_local, NextProg_local, Descendant_local,
                                   smass_slice, mcold_slice, mbh_slice, zcold_slice, sfrinst_slice, sfravg_slice,
                                   SnapNum_local, feature_slices, mass_node_cut, min_nodes)
                    if g is not None:
                        pickle.dump(g, fout, protocol=pickle.HIGHEST_PROTOCOL)
                        count_graphs += 1
                        saved_in_group += 1
                        if count_graphs % log_every_graphs == 0:
                            print(f"Guardados {count_graphs} grafos...")

                print(f" -> Guardados {saved_in_group} grafos en {gname}")

                # guardar estado después de cada grupo
                with open(state_file, "w") as sf:
                    json.dump({"done_idx": gi + 1, "count": count_graphs}, sf)

        print("\nFase 3 completada.")
        print("Total grafos guardados:", count_graphs)
        print("Output:", output_file)

if __name__ == "__main__":
    main()
