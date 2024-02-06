import math
import os
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
import random
import pandas as pd
import matplotlib.pyplot as plt

# Example data (replace these with your actual data)
def plot_graph(neighbors, accuracy1, accuracy2):
    # Plotting the graphs
    plt.figure(figsize=(8, 6))  # Adjust figure size if needed

    plt.plot(neighbors, accuracy1, marker='o', label='KNN using functions')
    plt.plot(neighbors, accuracy2, marker='s', label='KNN using sklearn')

    plt.title('Neighbors vs Accuracy')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()

# Find euclidean distance between two rows
def euclidean_distance(row1, row2):
    dist = 0.0
    for r in range(len(row2)):
        dist += (row1[r] - row2[r]) ** 2
    return math.sqrt(dist)

def most_occuring_elements(input_list):
    counts = Counter(tuple(item) if isinstance(item, list) else item for item in input_list)
    max_count = max(counts.values())  # Find the maximum count
    
    most_common_elements = [elem for elem, count in counts.items() if count == max_count]
    
    return most_common_elements

def K_nearest_neighbors(dataset, test_row, num_neighbors):
    distances = []
    for r in range(len(dataset)):
        row = dataset[r]
        # # This row is for dataset in manual
        # row1 = row[0:len(row)-1]
        # This row is for dataset in CSV
        row1 = row[1:len(row)]
        
        dis = euclidean_distance(row1, test_row)
        # # This d is for dataset in manual
        # d = [dis, row[len(row)-1:len(row)]]
        # The given d is from dataset in CSV
        d = [dis, row[0:1]]
        distances.append(d)
    
    distances.sort(reverse=False)
    
    k_nearest_labels = [label for _, label in distances[:num_neighbors]]  # Get labels of k nearest neighbors
    max_occuring_class = most_occuring_elements(k_nearest_labels)  # Find the most occuring class
    if len(max_occuring_class) > 1:
        return [random.choice(max_occuring_class)]
    return max_occuring_class   

# Data
# dataset = [[2.7810836,2.550537003,0],[1.465489372, 2.362125076,0],[3.396561688, 4.400293529,0],[1.38807019, 1.850220317,0],
#            [3.06407232,3.005305973,0],[7.627531214,2.759262235,1],[5.332441248, 2.088626775,1],[6.92259671, 1.771063677,1],
#            [8.675418651,-0.242068655,1], [7.673756466,3.508563011,1]]

def main():
    os.system('clear')
    neighbors_list = []
    acc1 = []
    acc2 = []
    for r in range(2,25):
        neighbors_list.append(r)
        neighbors = r
        dataset = pd.read_csv('/home/sami/Downloads/Sem 7/AI/Lab 09/fruit_data_with_colours.csv')

        dataset1 = dataset.values.tolist()
        orig_lbls = []
        pred_lbls = []
        for r in range(len(dataset1)):
            row = dataset1[r]
            # # Datasets in manual
            # row1 = row[0:len(row)-1]
            # orig_lbls.append(row[len(row)-1:len(row)])
            # pred_lbls.append(K_nearest_neighbors(dataset, row1, 5))

            # Dataset in CSV
            test_row = row[1:len(row)]
            orig_lbl = str(row[0:1])
            orig_lbl = orig_lbl.replace('[', '')
            orig_lbl = orig_lbl.replace(']', '') 
            orig_lbls.append(orig_lbl)
            pred_lbl = K_nearest_neighbors(dataset1, test_row, neighbors)
            pred_lbl = str(pred_lbl)
            pred_lbl = pred_lbl.replace('[', '')
            pred_lbl = pred_lbl.replace(']', '')
            pred_lbl = pred_lbl.replace(",", '')
            pred_lbl = pred_lbl.replace(")", '')
            pred_lbl = pred_lbl.replace("(", '')
            pred_lbls.append(pred_lbl)


        # Convert dataset to pandas DataFrame
        df = pd.DataFrame(dataset)

        # Select first column as label
        X = df.iloc[:, 1:]
        y = df.iloc[:, 0]

        # Create KNN classifier object
        knn = KNeighborsClassifier(n_neighbors=neighbors)

        # Train the model
        knn.fit(X, y)

        # Predict the labels for the dataset
        predicted_labels = knn.predict(X)

        # Calculate accuracy
        accuracy = sum(predicted_labels == y) / len(y) * 100
        acc2.append(accuracy)
        
        correct = 0
        for r in range(len(orig_lbls)):
            if orig_lbls[r] == pred_lbls[r]:
                correct += 1
        accuracy = correct / len(orig_lbls) * 100
        acc1.append(accuracy)
    
    plot_graph(neighbors_list, acc1, acc2)
if '__main__' == __name__:
    main()