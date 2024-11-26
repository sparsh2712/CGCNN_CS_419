import csv
import random

def shuffle_csv(csv_path):
    with open(csv_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader) 
        rows = list(reader)    

    random.shuffle(rows)

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)   

    print(f"Shuffled data written to {csv_path}")

if __name__ == '__main__':
    csv_path = "/Users/sparsh/Desktop/College core/CS_419/CGCNN_CS_419/data/id_prop.csv"
    shuffle_csv(csv_path)
