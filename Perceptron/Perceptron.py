import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

def preceptron(features, types, epoch = 100, bias = 0,decision_limit = 0.5, learning_rate = 0.1, accu_flg = False):
    X = features
    y = types

    w = np.zeros(len(X[0]))
    b = bias
    lr = learning_rate
    decision_limit = 0.5
    steps = 0
    accuracy = 0
    while (steps <= epoch) and (accuracy < 1):
        for j in range(len(X)): 
            o = np.dot(X[j], w) + b
            if o == y[j]:
                continue
            else:
                error =  (y[j] - o)*lr
                for i in range(len(X[j])):
                    w[i] += X[j][i] * error
            
        y_pred = []
        for i in range(len(X)):
            if np.dot(X[i], w) + b > decision_limit:
                y_pred.append(1)
            else:
                y_pred.append(0)
        acc = np.sum(y_pred == y) / len(y)
        if acc > accuracy:
            accuracy = acc
            w_best = w
            b_best = b
        if not accu_flg:
            steps += 1
    return w_best, b_best

def predict(test_data, w, b, decision_limit = 0.5):
    X = test_data
    y_pred = []
    for i in range(len(X)):
        if np.dot(X[i], w) + b > decision_limit:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred

def unique_classes(data):
    diff_classes = np.unique(data)
    element_map = {value: idx for idx, value in enumerate(diff_classes)}
    mapped_data = np.array([element_map[element] for element in data])
    return mapped_data

# train_data = np.array([[0, 0, 1], 
#                            [0, 1, 1], 
#                            [1, 0, 1], 
#                            [1, 1, 0]
#                            ]
#                         )
# x_train = train_data[:, 0:-1]
# y_train = train_data[:, -1]
# w, b = preceptron(features=x_train, types= y_train, learning_rate = 0.1, epoch = 1000, bias = 1, accu_flg = True)
# test_data = train_data[:, 0:-1]
# test_true = train_data[:, -1]
# test_pred = predict(test_data, w, b)
# score = accuracy_score(test_true, test_pred)*100
# print("Task_1 Accuracy: ",round(score,3),"%")
# data = pd.read_csv("/home/sami/Downloads/Sem 7/AI/Lab 12/sonar.all-data.csv")
# x = data.iloc[:,0:-1]
# y = data.iloc[:,-1]

# x= np.array(x)
# y= np.array(y)
# y = unique_classes(y)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=38)
# # Time start
# start = time.time()
# w, b = preceptron(features=x_train, types = y_train, learning_rate = 0.1, epoch = 1000, bias = 0.1)
# stop = time.time()
# print("Time taken: ",round(stop-start,3),"s")
# y_pred = predict(x_test, w, b)
# score = accuracy_score(y_test, y_pred)*100
# print(f"Task 2 Sonar Dataset")
# print("Accuracy : ",round(score,3), "%")

def main():
    os.system("clear")
    train_data = np.array([[0, 0, 1], 
                           [0, 1, 1], 
                           [1, 0, 1], 
                           [1, 1, 0]
                           ]
                        )
    x_train = train_data[:, 0:-1]
    y_train = train_data[:, -1]
    w, b = preceptron(features=x_train, types= y_train, learning_rate = 0.1, epoch = 1000, bias = 1, accu_flg = True)
    test_data = train_data[:, 0:-1]
    test_true = train_data[:, -1]
    test_pred = predict(test_data, w, b)
    score = accuracy_score(test_true, test_pred)*100
    print("Task_1 Accuracy: ",round(score,3),"%")

    data = pd.read_csv("/home/sami/Downloads/Sem 7/AI/Lab 12/sonar.all-data.csv")
    x = data.iloc[:,0:-1]
    y = data.iloc[:,-1]
    
    x= np.array(x)
    y= np.array(y)

    y = unique_classes(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=38)
    # Time start
    start = time.time()
    w, b = preceptron(features=x_train, types = y_train, learning_rate = 0.1, epoch = 1000, bias = 0.1)
    stop = time.time()
    print("Time taken: ",round(stop-start,3),"s")
    y_pred = predict(x_test, w, b)
    score = accuracy_score(y_test, y_pred)*100
    print(f"Task 2 Sonar Dataset")
    print("Accuracy : ",round(score,3), "%")

if __name__ == "__main__":
    main() 