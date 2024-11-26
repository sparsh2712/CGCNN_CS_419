from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
import torch 
import numpy as np 
import json 
import pandas as pd
from functools import cache
import os 
from pymatgen.core import Structure

def train_validate_test_loader(dataset, batch_size, train_ratio=None, val_ratio=0.1, test_ratio=0.1, 
                               collate_fn=default_collate, train_size=None, test_size=None, val_size=None):
    total_size = len(dataset)
    if not train_ratio:
        train_ratio = 1 - (test_ratio + val_ratio)
    
    indices = list(range(total_size))
    train_size = train_size if train_size else int(train_ratio * total_size)
    test_size = test_size if test_size else int(test_ratio * total_size)
    val_size = val_size if val_size else int(val_ratio * total_size)

    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(indices[train_size:train_size + val_size])
    test_sampler = SubsetRandomSampler(indices[-test_size:])

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=test_sampler, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader



def collate_pool(dataset_list):
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx, crystal_atom_idx, batch_target, batch_cif_ids = [], [], [], [], [], []
    base_idx = 0
    
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id) in enumerate(dataset_list):
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)

        batch_target.append(target)
        batch_cif_ids.append(cif_id)

        base_idx += n_i

    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx), torch.stack(batch_target, dim=0), batch_cif_ids

            

def gaussian_basis_transform(distances, dmin, dmax, step, var=None):
    filter = np.arange(dmin, dmax + step, step)
    var = step if var is None else var
    distances = np.expand_dims(distances, axis=-1)
    return np.exp(-(distances - filter) ** 2 / var ** 2)

class AtomicData:
    def __init__(self, path_to_atom_feature):
        with open(path_to_atom_feature, 'r') as file:
            data = json.load(file)

        self.atomic_embedding = {int(key): np.array(value, dtype=float) for key, value in data.items()}

    def get_atomic_data_features(self, atom):
        return self.atomic_embedding[atom]

class CIFDataset(Dataset):
    def __init__(self, cif_dir, atom_json_path, id_prop_path, max_neighbour=12, cutoff_radius=8, dmin=0, step=0.1):
        self.cif_dir, self.max_neighbour, self.cutoff_radius = cif_dir, max_neighbour, cutoff_radius

        id_prop_data = pd.read_csv(id_prop_path)
        self.id_prop_data = id_prop_data[['material_id', 'property_value']]
        self.atomic_data = AtomicData(atom_json_path) 
        self.dmin = dmin
        self.step = step

    def __len__(self):
        return len(self.id_prop_data)
    
    @cache 
    def __getitem__(self, index):
        cif_id, property_value = self.id_prop_data.loc[index, ['material_id', 'property_value']]
        cif_path = os.path.join(self.cif_dir, f'{cif_id}.cif')
        crystal = Structure.from_file(cif_path)
        all_neighbours = crystal.get_all_neighbors_py(self.cutoff_radius)
        all_neighbours = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_neighbours]

        neighbour_index = []
        neighbour_dist = []

        for atom_nbr in all_neighbours:
            if len(atom_nbr) < self.max_neighbour:
                print('not enough neighbours')
                neighbour_index.append(list(map(lambda x: x[2], atom_nbr)) + [0] * (self.max_neighbour - len(atom_nbr)))
                neighbour_dist.append(list(map(lambda x: x[1], atom_nbr)) + [self.radius + 1.] * (self.max_neighbour - len(atom_nbr)))
            else:
                neighbour_index.append(list(map(lambda x: x[2], atom_nbr[:self.max_neighbour])))
                neighbour_dist.append(list(map(lambda x: x[1], atom_nbr[:self.max_neighbour])))
        
        neighbour_index, neighbour_dist = np.array(neighbour_index), np.array(neighbour_dist)
        neighbour_dist = gaussian_basis_transform(neighbour_dist, self.dmin, self.cutoff_radius, self.step)

        atomic_features = np.vstack([self.atomic_data.get_atomic_data_features(crystal[i].specie.number) for i in range (len(crystal))])
        atomic_features = torch.Tensor(atomic_features)
        neighbour_dist = torch.Tensor(neighbour_dist)
        neighbour_index = torch.LongTensor(neighbour_index)
        target = torch.Tensor([float(property_value)])

        return (atomic_features, neighbour_dist, neighbour_index), target, cif_id
    
if __name__ == "__main__":
    cif = CIFDataset(cif_dir='/Users/sparsh/Desktop/College core/CS_419/CGCNN_CS_419/data/cif_files', atom_json_path= '/Users/sparsh/Desktop/College core/CS_419/CGCNN_CS_419/data/atom_init.json', id_prop_path='/Users/sparsh/Desktop/College core/CS_419/CGCNN_CS_419/data/id_prop.csv')
    print(cif[1]) 






    



    

    