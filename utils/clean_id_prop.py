import pandas as pd
import numpy as np 

def clean_data(propety_name):
    # Load the CSV file
    csv_file = '/Users/sparsh/Desktop/College core/CS_419/CGCNN_CS_419/data/id_prop.csv'
    df = pd.read_csv(csv_file)

    # Drop rows where 'target_value' is NaN or empty
    df = df[df['property_value'].notna()]  # Removes rows with NaN
    df = df[df['property_value'] != '']    # Removes rows with empty strings, if applicable

    df['property_value'] = pd.to_numeric(df['property_value'], errors='coerce')
    # Remove rows where 'target_value' is negative
    df = df[df['property_value'] >= 0]

    if propety_name in ['bulk_modulus', 'shear_modulus']:
        df = df[df['property_value']<=1000]
        df['property_value'] = np.log10(df['property_value'])

    df = df [['material_id', 'property_value']]
    # Save the cleaned DataFrame back to a CSV file
    df.to_csv('/Users/sparsh/Desktop/College core/CS_419/CGCNN_CS_419/data/id_prop.csv', index=False)

if __name__ == '__main__':
    clean_data('shear_modulus')
