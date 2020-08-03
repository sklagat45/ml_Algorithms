# P15/101552/2017
# SAMUEL KIPLAGAT RUTTO

import math
import operator
import random
import unicodecsv


# function to obtain data from csv file
def getdata(filename):
    with open(filename, 'rb') as f:
        reader = unicodecsv.reader(f)
        return list(reader)


# random train test data split function definition
def randomize(i_data):
    random.shuffle(i_data)
    train_data = i_data[:int(0.7*30)]
    test_data = i_data[int(0.7*30):]
    return train_data, test_data


def euclidean_dist(x, xi):
    d = 0.0
    for i in range(len(x)-1):
        d += pow((float(x[i])-float(xi[i])), 2)  # euclidean distance
    d = math.sqrt(d)
    return d


# KNN prediction and model training
def prediction(data_test, training_data, value_of_k):
    for i in data_test:
        euclid_dist =[]
        groups = []
        good = 0

        bad = 0
        for j in training_data:
            eu_dist = euclidean_dist(i, j)
            euclid_dist.append((j[5], eu_dist))
            euclid_dist.sort(key=operator.itemgetter(1))
            groups = euclid_dist[:value_of_k]
            for k in groups:
                if k[0] == 'g':
                    good += 1
                else:
                    bad +=1
        if good > bad:
            i.append('g')
        elif good < bad:
            i.append('b')
        else:
            i.append('NaN')


# Accuracy calculation function
def accuracy(test_data):
    correct = 0
    for i in test_data:
        if i[5] == i[6]:
            correct += 1
    accuracy = float(correct)/len(test_data) * 100
    return accuracy


data_set = getdata('iris_data_sample.csv')  # getdata function call with csv file as parameter
training_set, testing_set = randomize(data_set)  # train test data split
K = 2                                          # Assumed K value
prediction(testing_set, training_set, K)
print(testing_set)
print("Accuracy : ", accuracy(testing_set))
