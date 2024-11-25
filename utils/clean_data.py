import pandas as pd

with open('/Users/sparsh/Desktop/College core/CS_419/CGCNN_CS_419/data/cif_not_found.txt', 'r') as f:
    id_list = [line.strip() for line in f]

df = pd.read_csv('/Users/sparsh/Desktop/College core/CS_419/CGCNN_CS_419/data/material_ids_with_prop.csv')
print(len(df))
filtered_df = df[~df['material_id'].isin(id_list)]
print(len(filtered_df))
filtered_df = filtered_df[['material_id', 'formula_pretty', 'energy_per_atom', 'formation_energy_per_atom', 'band_gap', 'efermi', 'is_metal', 'bulk_modulus', 'shear_modulus', 'homogeneous_poisson']]
filtered_df['numeric_ids'] = filtered_df['material_id'].str.extract(r'mp-(\d+)').astype(int)
df_sorted = filtered_df.sort_values(by='numeric_ids').drop(columns='numeric_ids')
df_sorted.to_csv('/Users/sparsh/Desktop/College core/CS_419/CGCNN_CS_419/data/clean_material_ids_with_prop.csv')
