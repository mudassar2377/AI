import numpy as np
import math
import os
import pandas as pd
from sklearn.metrics import accuracy_score


#  Create a Baysian class
def baysian_classifier(data, split= 0.8, test_sample = None):
    
    #  Split the data into training and testing
    if test_sample is None:
        train_data = data.sample(frac=split, random_state=0)
        test_data = data.drop(train_data.index)

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values
    else:
        X_train = data.iloc[:, :-1].values
        y_train = data.iloc[:, -1].values
    
    #  Calculate the prior probabilities
    prior_prob = {}
    for i in range(len(y_train)):
        if y_train[i] not in prior_prob:
            prior_prob[y_train[i]] = 1
        else:
            prior_prob[y_train[i]] += 1

    for key in prior_prob:
        prior_prob[key] /= len(y_train)

    #  Calculating mean and variences
    mean = {}
    variance = {}
    for key in prior_prob:
        mean[key] = []
        variance[key] = []
        for i in range(len(X_train[0])):
            mean[key].append(np.mean(X_train[y_train == key, i]))
            variance[key].append(np.var(X_train[y_train == key, i]))
    
   
    if test_sample is None:
        #  Predict the class
         #  Calculation for posterior probabilities
        posterior_prob = {}
        for key in prior_prob:
            posterior_prob[key] = []
            for i in range(len(X_test)):
                posterior_prob[key].append(prior_prob[key])
                for j in range(len(X_test[0])):
                    posterior_prob[key][i] *= (1 / (math.sqrt(2 * math.pi * variance[key][j]))) * math.exp(-((X_test[i][j] - mean[key][j]) ** 2) / (2 * variance[key][j]))
        y_pred = []
        for i in range(len(X_test)):
            max_prob = 0
            predicted_class = None
            for key in posterior_prob:
                if posterior_prob[key][i] > max_prob:
                    max_prob = posterior_prob[key][i]
                    predicted_class = key
            y_pred.append(predicted_class)

        #  Calculate the accuracy
        score = accuracy_score(y_true=y_test, y_pred=y_pred)
        return score
    else:
        #  Calculation for posterior probabilities
        posterior_prob = {}
        for key in prior_prob:
            posterior_prob[key] = []
            posterior_prob[key].append(prior_prob[key])
            for j in range(len(test_sample)):
                posterior_prob[key][0] *= (1 / (math.sqrt(2 * math.pi * variance[key][j]))) * math.exp(-((test_sample[j] - mean[key][j]) ** 2) / (2 * variance[key][j]))
        
        # Predict class of test sample
        max_prob = 0
        predicted_class = None
        for key in posterior_prob:
            if posterior_prob[key][0] > max_prob:
                max_prob = posterior_prob[key][0]
                predicted_class = key
        return predicted_class
def main():
    # os.system('clear')
    # path = '/home/sami/Downloads/Sem 7/AI/Lab 10/diabetes.csv'
    # data = pd.read_csv(path, header=None)
    # # For Accuracy
    # score = baysian_classifier(data=data,split=0.5)
    # print("Task # 1 Finding Accuracy")
    # print("Accuracy: ",score)

    # Task 2
    data = [[12.5,6,0], [13,6.5,0], [13.5,7.5,0],
            [14,7,1], [14.5,8,1], [15,9,1],[15.5,8.5,1]]
    data = pd.DataFrame(data)
    test_sample = [13.75,8]
    # x = np.array([1, 0.5, 2])
    # y = np.array([2, 1.5, 3])


    
    # # Calculate covariance
    # covariance_matrix = np.cov(x, y)
    # print(covariance_matrix)
    # covariance_value = covariance_matrix[0, 1]
    # print(covariance_value)
    predicted_class = baysian_classifier(data=data,test_sample=test_sample)
    
    print("\n #--------------------------------------------------------------------------------------#\n\nTask # 2 Predicting Class")
    print("Predicted Class: ",predicted_class)

if __name__ == "__main__":
    main()