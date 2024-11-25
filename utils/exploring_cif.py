from pymatgen.core import Structure
import json 
import numpy as np 


def cif_processing(cif_path):
    crystal = Structure.from_file(cif_path)
    all_neighbours = crystal.get_all_neighbors_py(8)
    all_neighbours = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_neighbours]
    # print(all_neighbours)
    neighbour_index = []
    neighbour_dist = []
    for nbr in all_neighbours:
        neighbour_index.append(list(map(lambda x: x[2], nbr[:12])))
        neighbour_dist.append(list(map(lambda x: x[1], nbr[:12])))
    
    print(neighbour_index)



if __name__ == '__main__':
    cif_path = '/Users/sparsh/Desktop/College core/CS_419/CGCNN_CS_419/data/cif_files/mp-8.cif'
    cif_processing(cif_path)
