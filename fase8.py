import pickle
import random
from tqdm import tqdm
from torch_geometric.data import Data


dataset_train_path = "../data_fix/dataset_fases_train.pkl"

dataset_train_list = []
with open(dataset_train_path, "rb") as f:
    while True:
        try:
            g = pickle.load(f)
            if isinstance(g, Data):
                dataset_train_list.append(g)
        except EOFError:
            break

print(f"Total grafos cargados train: {len(dataset_train_list)}")
print("Ejemplo shape x train:", dataset_train_list[0].x.shape)
print("Ejemplo y train:", dataset_train_list[0].y)
