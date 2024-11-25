import pandas as pd
import numpy as np
from mp_api.client import MPRester

def fetch_cif_files(api_key, cif_list_path, output_directory):
    """
    Fetches CIF files for materials listed in the provided CSV file.

    Args:
    - api_key (str): The API key for accessing the Materials Project API.
    - cif_list_path (str): Path to the CSV file containing material IDs.
    - output_directory (str): Directory where CIF files will be saved.

    Returns:
    - None
    """

    m = MPRester(api_key=api_key)
    cif_list = pd.read_csv(cif_list_path, header=None, names=['material-id'])
    cif_list['material-id'] = cif_list['material-id'].astype(str)

    for material_id in cif_list['material-id']:
        try:
            structure = m.get_structure_by_material_id(material_id)
            cif_data = structure.to(fmt="cif")
            with open(f'{output_directory}/{material_id}.cif', 'w') as f:
                f.write(cif_data)
        except Exception as e:
            with open('cif_not_found.txt', 'a') as t:
                t.write(f'{material_id}\n')
            print(f"Error occurred for material ID {material_id}: {e}")

if __name__ == '__main__':
    api_key = "PULAwCzMQFzgXRhElkO1T4BoM6mQAjnO"
    cif_list_path = '/Users/sparsh/Desktop/College core/CS_419/CGCNN_CS_419/data/mp-ids-3402.csv'
    output_directory = '/Users/sparsh/Desktop/College core/CS_419/CGCNN_CS_419/data/cif_files'

    fetch_cif_files(api_key, cif_list_path, output_directory)
