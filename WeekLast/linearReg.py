import csv
from sklearn.svm import SVC
import numpy as np
import math
import random

firstVal = 9.0
# secondVal = 5.0

# Turn transformation on or off
transformD = 1
lamb = 1

def transform(points):
	transformed = []
	for point in points:
		x1 = point[0]
		x2 = point[1]
		newLine = [1, x1, x2, x1*x2, math.pow(x1, 2), math.pow(x2, 2)]
		# newLine = [1, x2, x2]
		transformed.append(newLine)
	return transformed


xDataTrain = []
yDataTrain = []

with open('features.train.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		if(float(row[0]) == firstVal):
			yDataTrain.append([1])
		else:
			yDataTrain.append([-1])
		xData = []
		for i in row[1:]:
			xData.append(float(i))
		xDataTrain.append(xData)

		# This is for if there is a firstVal and secondVal.
		# xData = []
		# for i in row[1:]:
		# 	xData.append(float(i))
		# if(float(row[0]) == firstVal):
		# 	yDataTrain.append([1])
		# 	xDataTrain.append(xData)
		# elif(float(row[0]) == secondVal):
		# 	yDataTrain.append([-1])
		# 	xDataTrain.append(xData)

xDataTest = []
yDataTest = []

with open('features.test.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		if(float(row[0]) == firstVal):
			yDataTest.append([1])
		else:
			yDataTest.append([-1])
		xData = []
		for i in row[1:]:
			xData.append(float(i))
		xDataTest.append(xData)

		# This is for if there is a firstVal and secondVal.
		# xData = []
		# for i in row[1:]:
		# 	xData.append(float(i))
		# if(float(row[0]) == firstVal):
		# 	yDataTest.append([1])
		# 	xDataTest.append(xData)
		# elif(float(row[0]) == secondVal):
		# 	yDataTest.append([-1])
		# 	xDataTest.append(xData)

if(transformD):
	xDataTrain = transform(xDataTrain)
	xDataTest = transform(xDataTest)

#Linear Regression with Weight Decay
xTrain = np.matrix(xDataTrain)
yTrain = np.matrix(yDataTrain)
xTest = np.matrix(xDataTest)
yTest = np.matrix(yDataTest)

identitySize = len(xDataTrain[0])
innerResult = (np.transpose(xTrain) * xTrain) + lamb*np.identity(identitySize)
weight = np.linalg.inv(innerResult) * np.transpose(xTrain) * yTrain

Ein = 0
for i in range(len(xDataTrain)):
	point = np.matrix(xDataTrain[i])
	result = point * weight
	if(np.sign(result) != yTrain[i]):
		Ein += 1

Eout = 0
for i in range(len(xDataTest)):
	point = np.matrix(xDataTest[i])
	result = point * weight
	if(np.sign(result) != yTest[i]):
		Eout += 1

# print("First Val: " + str(firstVal))
print("Ein: " + str(Ein/float(len(xDataTrain))))
print("Eout: " + str(Eout/float(len(xDataTest))))
# print("Size of training data: " + str(len(xDataTrain)))
# print("Size of testing data: " + str(len(xDataTest)))
