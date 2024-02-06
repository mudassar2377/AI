import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score

def gaussian_pdf(x, mean, std):
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean)**2 / (2 * std**2)))

def naive_bayes(dataset, test_ratio = 0.2, test_sample = None):
    # Splitting the dataset into the Training set and Test set
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    if test_sample is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=0)
    else:
        X_train = X
        y_train = y
        X_test = test_sample
        y_test = None
    # Prior Probabilities
    prior_prob = {}
    prior_prob[0] = np.count_nonzero(y_train == 0) / len(y_train)
    prior_prob[1] = np.count_nonzero(y_train == 1) / len(y_train)

    # Calculating mean and standard deviation for each feature
    mean = {}
    std = {}
    for i in range(len(X_train[0])):
        mean[i] = {}
        std[i] = {}
        for j in range(2):
            mean[i][j] = np.mean(X_train[y_train == j, i])
            std[i][j] = np.std(X_train[y_train == j, i])
    
    # Predicting the Test set results 
    y_pred = []
    for i in range(len(X_test)):
        prob = {}
        for j in range(2):
            prob[j] = prior_prob[j]
            for k in range(len(X_test[0])):
                prob[j] *= gaussian_pdf(X_test[i][k], mean[k][j], std[k][j])
        y_pred.append(max(prob, key=prob.get))
    
    if test_sample is not None:
        return y_pred
    else:
        # Calculating accuracy
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy    

def main():
    os.system('clear')
    dataset = [[25, 40000, 0],
               [35, 60000, 0],
               [45, 80000, 0],
               [20, 20000, 0],
               [35, 120000, 0],
               [52, 18000, 0],
               [23, 95000, 1],
               [40, 62000, 1],
               [60, 100000, 1],
               [48, 22000, 1],
               [33, 150000, 1]]
    dataset = pd.DataFrame(dataset)
    print(f'Task 1 : Manual Dataset for class prediction')
    pred_class = naive_bayes(dataset=dataset, test_sample=[[40, 20000]])
    print(f'Predicted Class : {pred_class}')
    print(f'\n\n %---------------------------------------------------------%\n\n\nTask 2 : Diabetes Dataset Accuracy')

    # Importing the dataset
    dataset = pd.read_csv('/home/sami/Downloads/Sem 7/AI/Lab 10/diabetes.csv')
    score = naive_bayes(dataset=dataset,test_ratio= 0.5)
    
    print("Accuracy : ", score)

    
    
if __name__ == '__main__':
    main()