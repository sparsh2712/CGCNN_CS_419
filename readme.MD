# Crystal Graph Convolutional Neural Network (CGCNN)

## Workflow

### Data Preparation
```bash
python utils/feature_extraction.py
python utils/clean_id_prop.py
python utils/shuffle_id_prop.py
```

### Training and Testing 
```bash
python main.py
```

### Results Visualization 
```bash
python utils/create_graph.py
```

## Repository Structure

- `data.py`: Contains data utilities and dataset class for loading and handling data.
- `cgcnn.py`: Main implementation of the Crystal Graph Convolutional Neural Network (CGCNN).
- `main.py`: Wrapper script for training and running the model.

### `utils/`
Contains utility scripts for various tasks related to data collection, preprocessing, and analysis.

- `cif.py`: Functions to download CIF files.
- `clean_data.py`: Selects entries where both CIF and property data are present.
- `clean_id_prop.py`: Cleans the `id_prop` table by removing outliers and random values.
- `create_graph.py`: Generates plots for model predictions versus true values, calculates Mean Absolute Error (MAE) vs epoch, and evaluates the R² score.
- `data_download.py`: Script to download property data for training.
- `exploring_cif.py`: Explores how to work with CIF files using `pymatgen`.
- `shuffle_id_prop.py`: Shuffles the `id_prop` data for randomization in training.

### `models/`
Contains the best-performing models saved during training.

### `results/`
Stores the results of training and evaluation, including metrics, plots, and logs.