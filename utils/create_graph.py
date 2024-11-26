import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def calculate_r2(csv_path):
    data = pd.read_csv(csv_path, header=None, names=['actual', 'predicted'])
    actual = data['actual']
    predicted = data['predicted']
    r2 = r2_score(actual, predicted)
    return r2

def plot_actual_vs_predicted(csv_path, output_image_path):
    data = pd.read_csv(csv_path, header=None, names=['material_id', 'true_value', 'predicted_value'])
    true_values = data['true_value']
    predicted_values = data['predicted_value']
    plt.figure(figsize=(8, 8))
    plt.scatter(true_values, predicted_values, alpha=0.7, label='Data points')
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='red', linestyle='--', label='y=x (Ideal)')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()

def plot_epoch_vs_mae(file_path, output_image_path):
    epochs = []
    mae_errors = []
    
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('->')
            epoch = int(parts[0].split(':')[1].strip())
            mae_err = float(parts[1].split(':')[1].strip())
            epochs.append(epoch)
            mae_errors.append(mae_err)
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, mae_errors, marker='o', linestyle='-', label='MAE Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE Error')
    plt.title('Epoch vs MAE Error')
    plt.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_image_path)
    plt.close()


if __name__ == '__main__':
    plot_actual_vs_predicted('results/test_results.csv', 'results/graph.png')
    plot_epoch_vs_mae('results/val_mae_err.txt', 'results/epoch_vs_mae.png')
    print(f'R2 :{calculate_r2('results/test_results.csv')}')
