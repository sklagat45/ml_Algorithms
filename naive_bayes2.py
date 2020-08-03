# P15/101552/2017
# SAMUEL KIPLAGAT RUTTO
#Machine Learning Algorithms
#Naive-Bayes Algorithm
#University of Nairobi

import csv
import math
import random


def loadCsv(filename):
	#opens a csv file in read only mode
	lines = csv.reader(open(filename, "r"))
	#creates a lists by the name dataset from the csv file
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def splitDataset(dataset, splitRatio):
	#this creates two different sized arrays testing and training
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	#copies data from the dataset into the training set dependng on the spilt ratio
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

#separate the training dataset instances by class value so that we can calculate statistics for each class.
#this function creates vectors from the different rows of the dataset
def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

#calculate the mean of each attribute for a class value
def mean(numbers):
	return sum(numbers)/float(len(numbers))

#standard deviation describes the variation of spread of the data
#and we will use it to characterize the expected spread of each attribute
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

#zip function groups the values for each attribute across our data instances into their own lists
#this is done to compute the stdev and mean for the attribute
def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

#Separates the training dataset into instances grouped by class
#calculates summaries for each attribute after this
def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

#the attribute summaries where prepared for each attribute and class value
#the result is the conditional probability of a given attribute value given a class value
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

#this calculates the probability of an entire data instance by multiplying them
#the resul tis a map of class values to probabilities
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

#looks for the largest probability and return the associated class
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

#estimate the accuracy of the model by making predictions for each data instance in our test dataset
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

#The predictions can be compared to the class values in the test dataset and a classification accuracy can be calculated
def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
 
def main():
	filename = 'naivecsv2.csv'
	splitRatio = 0.67
	dataset = loadCsv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
	# prepare model
	summaries = summarizeByClass(trainingSet)
	# test model
	predictions = getPredictions(summaries, testSet)
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: {0}%'.format(accuracy))
 
main()