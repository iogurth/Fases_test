# fase4_reduction_streaming.py
"""
Fase 4: reducción tipo Mangrove (streaming).
Lee un PKL donde cada objeto fue volcado con pickle.dump(obj, f) (uno a uno),
reduce cada árbol manteniendo nodos relevantes (first-detection, pre-merge, post-merge, final),
y escribe los árboles reducidos a un nuevo PKL (append). Mantiene estado para reanudar.
"""

import pickle
import json
import os
from collections import Counter, defaultdict
import numpy as np
import torch
from tqdm import tqdm
import os

os.chdir("../../../../../../home/iogurth/gdrive/TNG300_MergerTrees")

# --------- Ajustes / paths ----------
INPUT_PKL = "dataset_fases/dataset_phase3_mangrove_like_all_targets_FULLRAW_FIX_extra_info.pkl"
OUTPUT_PKL = "dataset_fases/dataset_phase4_reduced_mangrove_like.pkl"
STATE_FILE = "dataset_fases/state_phase4.json"

# límites y parámetros
MAX_NODES_PER_TREE = 20000     # si el árbol reducido > esto -> lo descartamos (igual que paper)
MIN_NODES_PER_TREE = 10         # mínimo para guardar
VERBOSE = True

# --------- Helpers ----------
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"processed": 0, "saved": 0, "skipped": 0}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def is_array_like(x):
    return isinstance(x, (list, tuple, np.ndarray, torch.Tensor))

# --------- Reduction logic (Mangrove-like) ----------
def reduce_graph(g, max_nodes=MAX_NODES_PER_TREE, min_nodes=MIN_NODES_PER_TREE):
    """
    Entrada: g es un objeto torch_geometric.data.Data con al menos:
      - g.x (N, F)
      - g.edge_index (2, E)
      - g.subhalo_ids (N,)  -- SubhaloID por fila (recomendado)
      - g.snapnum (N,)      -- SnapNum alineado.
      - g.first_progenitor (N,) -- índice dentro del grafo (o -1)
      - g.next_progenitor (N,)  -- índice dentro del grafo (o -1)
      - g.descendant (N,)   -- índice dentro del grafo (o -1) o puede estar ausente

    Devuelve: Data reducido o None (si no cumple min/max)
    """
    # comprobar atributos
    attrs = {}
    for attr in ("subhalo_ids", "snapnum", "first_progenitor", "next_progenitor"):
        if not hasattr(g, attr):
            raise RuntimeError(f"El grafo no tiene atributo requerido: {attr}")
        attrs[attr] = getattr(g, attr).cpu().numpy() if torch.is_tensor(getattr(g,attr)) else np.array(getattr(g,attr))
    descendant_attr = None
    if hasattr(g, "descendant"):
        descendant_attr = getattr(g, "descendant")
        descendant_attr = descendant_attr.cpu().numpy() if torch.is_tensor(descendant_attr) else np.array(descendant_attr)

    N = g.x.shape[0]
    if N == 0:
        return None

    subhalo_ids = attrs["subhalo_ids"]
    snapnum = attrs["snapnum"].astype(int)
    first_prog = attrs["first_progenitor"].astype(int)
    next_prog = attrs["next_progenitor"].astype(int)
    descendant = descendant_attr.astype(int) if descendant_attr is not None else np.full_like(first_prog, -1)

    # Normalizar valores inválidos a -1
    first_prog[first_prog < 0] = -1
    next_prog[next_prog < 0] = -1
    descendant[descendant < 0] = -1

    # contar progenitores por cada posible descendiente (conteo en índice local)
    # progenitor_count[idx_desc] = número de nodos cuyo descendant == idx_desc
    progenitor_count = Counter()
    for i, d in enumerate(descendant):
        if d >= 0:
            progenitor_count[int(d)] += 1

    # identificar nodo "final" (máximo SnapNum) - se asume siempre presente
    final_idx = int(np.argmax(snapnum))

    # marcadores de nodos a mantener
    keep = np.zeros(N, dtype=bool)

    # criterio 1: nodos sin progenitores -> first detection
    keep[first_prog == -1] = True

    # criterio 2 y 3: pre-merge y post-merge
    # post-merge: nodos que tienen progenitor_count > 1 (es decir resultan de fusión)
    for idx, cnt in progenitor_count.items():
        if 0 <= idx < N and cnt > 1:
            keep[int(idx)] = True  # post-merge (descendiente con múltiples progenitores)
    # pre-merge: cualquier nodo cuyo descendant tiene >1 progenitores y cuyo snapnum == descendant_snapnum + 1
    for i in range(N):
        d = descendant[i]
        if d >= 0 and progenitor_count.get(int(d), 0) > 1:
            # ver si es justo un snapshot antes: snap(i) == snap(descendant) + 1  (descendant tiene snap menor)
            if snapnum[i] == snapnum[int(d)] + 1:
                keep[i] = True

    # criterio 4: conservar la raíz/final
    keep[final_idx] = True

    # (opcional) conservar nodos que tienen next_progenitor linkage (otros progenitores en su snapshot)
    # Esto puede ayudar a mantener ramas múltiples que existen en el mismo snapshot
    keep[next_prog != -1] = True

    # Si no hay suficientes nodos, devolvemos None
    n_kept = int(keep.sum())
    if n_kept < min_nodes:
        return None

    # Si supera el máximo permitido -> descartar (igual que Mangrove)
    if n_kept > max_nodes:
        return None

    # Construir nuevo Data reducido: filtrar x y atributos, reconstruir edges
    # obtener indices old->new
    old_to_new = {}
    new_idx = 0
    for old_i in range(N):
        if keep[old_i]:
            old_to_new[old_i] = new_idx
            new_idx += 1

    # filtrar x
    x_np = g.x.cpu().numpy() if torch.is_tensor(g.x) else np.array(g.x)
    Xc = torch.tensor(x_np[keep], dtype=torch.float32)

    # reconstruir edges (filtrar aristas internas entre nodos mantenidos)
    edge = g.edge_index
    if torch.is_tensor(edge):
        edges_np = edge.cpu().numpy().T  # (E,2)
    else:
        edges_np = np.array(edge).T
    new_edges = []
    for (u, v) in edges_np:
        if (u in old_to_new) and (v in old_to_new):
            uu = old_to_new[int(u)]
            vv = old_to_new[int(v)]
            if uu != vv:
                new_edges.append([uu, vv])
    if len(new_edges) == 0:
        return None
    new_edge_index = torch.tensor(np.array(new_edges).T, dtype=torch.long).contiguous()

    # atributos alineados
    def map_attr(arr):
        arr_np = arr.cpu().numpy() if torch.is_tensor(arr) else np.array(arr)
        return torch.tensor(arr_np[keep], dtype=torch.int64 if np.issubdtype(arr_np.dtype, np.integer) else torch.float32)

    subhalo_ids_new = map_attr(g.subhalo_ids)
    snapnum_new = map_attr(g.snapnum)
    # remap first_progenitor/next_progenitor/descendant a índices dentro del grafo reducido:
    def remap_link(arr):
        arr_np = arr.cpu().numpy() if torch.is_tensor(arr) else np.array(arr)
        out = np.full(int(keep.sum()), -1, dtype=np.int64)
        kept_idxs = np.nonzero(keep)[0]
        for out_pos, old_idx in enumerate(kept_idxs):
            val = int(arr_np[old_idx])
            out[out_pos] = old_to_new.get(val, -1)
        return torch.tensor(out, dtype=torch.long)

    first_prog_new = remap_link(g.first_progenitor)
    next_prog_new = remap_link(g.next_progenitor)
    descendant_new = remap_link(getattr(g, "descendant", np.full(N, -1)))

    # target y (si existe)
    y_new = g.y.clone() if hasattr(g, "y") else None

    # construir Data reducido
    data = type(g)() if isinstance(g, type(torch.utils.data.Dataset())) else torch.zeros(1)  # dummy type guard (we'll use Data)
    from torch_geometric.data import Data as GeomData
    data = GeomData(x=Xc, edge_index=new_edge_index, y=y_new)
    data.subhalo_ids = subhalo_ids_new
    data.snapnum = snapnum_new
    data.first_progenitor = first_prog_new
    data.next_progenitor = next_prog_new
    data.descendant = descendant_new

    return data

# --------- Streaming processing ----------
def process_stream(input_pkl=INPUT_PKL, output_pkl=OUTPUT_PKL):
    state = load_state()
    processed = state.get("processed", 0)
    saved = state.get("saved", 0)
    skipped = state.get("skipped", 0)

    # abrir archivos
    with open(input_pkl, "rb") as fin, open(output_pkl, "ab") as fout:
        # si reanudamos, descartamos (unpickle) los primeros `processed` objetos
        if processed > 0:
            if VERBOSE:
                print(f"Resumiendo: descartando {processed} objetos ya procesados...")
            for _ in tqdm(range(processed), desc="Skipping processed", leave=False):
                try:
                    _ = pickle.load(fin)
                except EOFError:
                    print("EOF mientras saltaba procesados: nada más que hacer.")
                    break

        idx = processed
        pbar = tqdm(desc="Reduciendo árboles (fase4)", unit="trees")
        while True:
            try:
                g = pickle.load(fin)
            except EOFError:
                break

            idx += 1
            # aplicar reducción
            try:
                reduced = reduce_graph(g)
                if reduced is not None:
                    pickle.dump(reduced, fout, protocol=pickle.HIGHEST_PROTOCOL)
                    saved += 1
                else:
                    skipped += 1
            except Exception as e:
                # si falla en una gráfica concreta, la saltamos (guardamos en skipped)
                skipped += 1
                if VERBOSE:
                    print(f"[WARN] Error reduciendo grafo idx={idx}: {e}")

            # actualizar estado y flush cada N
            if idx % 50 == 0:
                processed = idx
                state = {"processed": processed, "saved": saved, "skipped": skipped}
                save_state(state)

            pbar.update(1)

        # estado final
        processed = idx
        state = {"processed": processed, "saved": saved, "skipped": skipped}
        save_state(state)
        pbar.close()

    print("Fase 4 completada.")
    print("Procesados:", processed, "Guardados:", saved, "Saltados:", skipped)
    return state

if __name__ == "__main__":
    st = process_stream()
    print("Estado final:", st)
