import csv
from sklearn.svm import SVC
import numpy as np
import math

# These stand for how the data should be classified.
firstVal = 1.0
secondVal = 5.0
Cval = math.pow(10,6)
degreeVal = 5

# This classifies the test data using the firstVal and secondVal
xDataTest = []
yDataTest = []
with open('features.test.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		# This is for if there is a firstVal and secondVal.
		xData = row[1:]
		if(float(row[0]) == firstVal):
			yDataTest.append(1)
			xDataTest.append(xData)
		elif(float(row[0]) == secondVal):
			yDataTest.append(-1)
			xDataTest.append(xData)

		# This is for if there is only a firstVal, meaning firstVal vs. all

		# if(float(row[0]) == firstVal):
		# 	yDataTest.append(1)
		# else:
		# 	yDataTest.append(-1)
		# xData = row[1:]
		# xDataTest.append(xData)

# This classifies the train data using the firstVal and secondVal
xDataTrain = []
yDataTrain = []
with open('features.train.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile)
	for row in reader:
		# This is for if there is a firstVal and secondVal.
		xData = row[1:]
		if(float(row[0]) == firstVal):
			yDataTrain.append(1)
			xDataTrain.append(xData)
		elif(float(row[0]) == secondVal):
			yDataTrain.append(-1)
			xDataTrain.append(xData)

		# This is for if there is only a firstVal, meaning firstVal vs. all

		# if(float(row[0]) == firstVal):
		# 	yDataTrain.append(1)
		# else:
		# 	yDataTrain.append(-1)
		# xData = row[1:]
		# xDataTrain.append(xData)

print(xDataTrain)

# This creates the X and y array. It then initializes the clf with all of the
# parameters (C, degree, kernel, etc).
X = np.array(xDataTrain)
y = np.array(yDataTrain)
clf = SVC()
print('C = ' + str(Cval))

# For problems 1-6, kernel = 'poly'. For problems 9-10, kernel = 'rbf'.
clf.set_params(C = Cval, degree = degreeVal, kernel = 'rbf', coef0 = 1.0, gamma = 1)
clf.fit(X, y)

# This gets the number of support vectors.
numSupport = clf.support_vectors_
print('Num Support Vectors: ' + str(len(numSupport)))

#Checking Ein by predicting on the training data.
predictionIn = clf.predict(xDataTrain)
numErrorIn = 0
#Ein is computed by counting the number of misclassified data points.
for i in range(len(predictionIn)):
	if(predictionIn[i] != yDataTrain[i]):
		numErrorIn += 1
print('Error In: ' + str(numErrorIn/float(len(yDataTrain))))

#Checking Eout by predicting on the test data.
predictionOut = clf.predict(xDataTest)
numErrorOut = 0
#Eout is computed by counting the number of misclassified data points.
for i in range(len(predictionOut)):
	if(predictionOut[i] != yDataTest[i]):
		numErrorOut += 1
print('Error Out: ' + str(numErrorOut/float(len(yDataTest))))
