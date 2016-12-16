import csv
from sklearn.svm import SVC
import numpy as np
import math
import random

#This classifies the inputted point based on the classify function
def classification(x1, x2):
	val = x2 - x1 + (0.25 * math.sin(math.pi * x1))
	return np.sign(val)

# This function generates 100 points to train on
def generateTrainingPoints():
	points = []
	for i in range(100):
		points.append([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
	return points

# This function generates 500 points to test on
def generateTestingPoints():
	points = []
	for i in range(500):
		points.append([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])
	return points

# Initialize by choosing K random points as centroids
def initializeBins(points, K):
	bins = []
	centroids = []
	randomIndices = random.sample(range(len(points)), K)
	for index in randomIndices:
		bins.append([points[index]])
		centroids.append(points[index])
	return bins, centroids

# This gets the distance between two points using the distance formula
def getDistance(pt1, pt2):
	x1 = pt1[0]
	x2 = pt2[0]
	y1 = pt1[1]
	y2 = pt2[1]
	distance = math.sqrt(math.pow((x2-x1), 2) + math.pow((y2-y1), 2))
	return distance

# Fill the bins by iterating through all the points and then putting them in 
# the bins whose centroid is closest to them.
def fillBins(bins, centroids, points):
	for i in range(len(points)):
		curPoint = points[i]
		minDist = 10000000
		bestVal = -1
		for i in range(len(centroids)):
			dist = getDistance(curPoint, centroids[i])
			if dist < minDist:
				minDist = dist
				bestVal = i
		bins[bestVal].append(curPoint)

# This finds the median point in each bin and sets that as the new centroid.
def computeNew(bins, K):
	newBins = []
	newCentroids = []
	for i in range(K):
		curLst = bins[i]
		size = len(curLst)
		minX = curLst[0][0]
		maxX = curLst[0][0]
		minY = curLst[0][1]
		maxY = curLst[0][1]
		for x in range(1, size):
			if curLst[x][0] < minX:
				minX = curLst[x][0]
			if curLst[x][0] > maxX:
				maxX = curLst[x][0]
			if curLst[x][1] < minY:
				minY = curLst[x][1]
			if curLst[x][1] > maxY:
				maxY = curLst[x][1]
		avgX = (minX + maxX)/2
		avgY = (minY + maxY)/2
		avgPoint = [avgX, avgY]
		smallestDist = 100000
		bestPoint = curLst[0]
		for y in range(size):
			dist = getDistance(avgPoint, curLst[y])
			if dist < smallestDist:
				bestPoint = curLst[y]
		newBins.append([bestPoint])
		newCentroids.append(bestPoint)
	return newBins, newCentroids

# This compares two lists. Returns the number of differences
def compareLists(old, new):
	differences = 0
	for i in new:
		if i not in old:
			differences += 1
	return differences

# This generates the matrix needed for lloyd's algorithm
def generateMatrix(data, centroids, gamma, K):
	mat = []
	for i in range(len(data)):
		curPoint = data[i]
		mat.append([1])
		for j in range(K):
			val = -1 * gamma *  math.pow(getDistance(curPoint, centroids[j]), 2)
			mat[i].append(math.exp(val))
	return mat

# This runs the kernel form of RBF
def kernelForm(train, test, gam, K):
	clf = SVC()

	# Build the yMatrixTraining set
	yMatrixTrain = []
	for i in train:
		val = classification(i[0], i[1])
		yMatrixTrain.append(val)

	# Build the yMatrixTesting set
	yMatrixTest = []
	for i in test:
		val = classification(i[0], i[1])
		yMatrixTest.append(val)

	# Converts the inputted point arrays into np arrays
	X = np.array(train)
	y = np.array(yMatrixTrain)

	# Sets the parameters for SVM and fits them with the data
	clf.set_params(C = float('inf'), kernel = 'rbf', coef0 = 1.0, gamma = gam)
	clf.fit(X, y)

	# Predict on training data to find Ein
	predictionIn = clf.predict(train)
	numErrorIn = 0

	# Checks the amount of errors
	for i in range(len(predictionIn)):
		if(predictionIn[i] != yMatrixTrain[i]):
			numErrorIn += 1

	# Predict on the test data to find Eout
	predictionOut = clf.predict(test)
	numErrorOut = 0

	# Checks the amount of errors
	for i in range(len(predictionOut)):
		if(predictionOut[i] != yMatrixTest[i]):
			numErrorOut += 1

	# Computes and outputs the error
	numErrorIn = numErrorIn/float(len(train))
	numErrorOut = numErrorOut/float(len(test))
	return [numErrorIn, numErrorOut]

# This runs the regular form of RBF
def regularForm(train, test, gam, K):
	# Computes the yMatrixTraining Data
	yMatrixTrain = []
	for i in train:
		val = classification(i[0], i[1])
		yMatrixTrain.append([val])

	# Computes the yMatrixTesting Data
	yMatrixTest = []
	for i in test:
		val = classification(i[0], i[1])
		yMatrixTest.append(val)

	size = len(train)

	# Initializes K Clustering
	curBins, curCentroids = initializeBins(train, K)
	fillBins(curBins, curCentroids, train)
	newBins, newCentroids = initializeBins(train, K)

	# Runs K Clustering
	while(compareLists(curCentroids, newCentroids)):
		curBins = newBins
		curCentroids = newCentroids
		newBins, newCentroids = computeNew(curBins, K)
		fillBins(newBins, newCentroids, train)

	# Checks for empty bins
	noEmpty = 1
	for bin in curBins:
		if len(bin) == 0:
			noEmpty = 0

	# If no empty bins
	if(noEmpty):

		# Generate the matrix
		mat = generateMatrix(train, curCentroids, gam, K)
		innerMat = np.matrix(mat)
		# Find the pseudo inverse to calculate the weight
		pInv = np.linalg.pinv(innerMat)
		yMat = np.matrix(yMatrixTrain)
		weight = np.dot(pInv, yMat)

		# Gets the yMatrix for the training data
		yMatTrain = innerMat * weight
		trueY = []
		for x in np.nditer(yMatTrain):
			trueY.append(np.sign(x))
		# Computes Ein
		Ein = 0
		for i in range(len(trueY)):
			if(trueY[i] != yMatrixTrain[i]):
				Ein += 1

		# Computes Eout by getting the yMatrix for the testing data
		Eout = 0
		matTest = generateMatrix(test, curCentroids, gam, K)
		innerMatTest = np.matrix(matTest)

		yMatTest = innerMatTest * weight

		trueYtest = []
		for x in np.nditer(yMatTest):
			trueYtest.append(np.sign(x))
		for i in range(len(trueYtest)):
			if(trueYtest[i] != yMatrixTest[i]):
				Eout += 1

		# Computes the error and outputs
		Ein = Ein/float(len(train))
		Eout = Eout/float(len(test))
		return [Ein, Eout]

	else:
		return -1

# Variables for the simulation

k = 9
gam = 1.5

kernelEin = 0
kernelEout = 0

kernelBetter = 0
regularBetter = 0

numNonSep = 0
numEin0 = 0

runs = 0
numRealRuns = 0
numRuns = 100

# For question 16 and 17
optA = 0
optB = 0
optC = 0
optD = 0
optE = 0

while(runs < numRuns):

	points = generateTrainingPoints()
	testPts = generateTestingPoints()

	# ----- This Code is for 14/15 ----------
	# kernelResult = kernelForm(points, testPts, gam, k)
	# if(kernelResult[0] == 0):
	# 	regResult = regularForm(points, testPts, gam, k)
	# 	if(regResult != -1):

	# 		# Compare the EOuts
	# 		numRealRuns += 1
	# 		if(kernelResult[1] > regResult[1]):
	# 			regularBetter += 1
	# 		elif(regResult[1] > kernelResult[1]):
	# 			kernelBetter += 1

	# ------ This code is for 13 ------------
	kernelResult = kernelForm(points, testPts, gam, k)
	kernelEin += kernelResult[0]
	kernelEout += kernelResult[1]

	# if(kernelResult[0] != 0):
	# 	numNonSep += 1

	# ----- This code is for 16, 17 -----
	# --- For 16
	# regOld = regularForm(points, testPts, 1.5, 9)
	# regNew = regularForm(points, testPts, 1.5, 12)

	# --- For 17
	# regOld = regularForm(points, testPts, 1.5, 9)
	# regNew = regularForm(points, testPts, 2, 9)

	# if(regOld != -1 and regNew != -1):
	# 	numRealRuns += 1

	# 	#If Ein goes down, but Eout goes up
	# 	if(regNew[0] < regOld[0] and regNew[1] > regOld[1]):
	# 		optA += 1
	# 	# If Ein goes up, but Eout goes down
	# 	if(regNew[0] > regOld[0] and regNew[1] < regOld[1]):
	# 		optB += 1
	# 	# If both go up
	# 	if(regNew[0] > regOld[0] and regNew[1] > regOld[1]):
	# 		optC += 1
	# 	# If both go down
	# 	if(regNew[0] < regOld[0] and regNew[1] < regOld[1]):
	# 		optD += 1
	# 	# If both remain the same
	# 	if(regNew[0] == regOld[0] and regNew[1] == regOld[1]):
	# 		optE += 1

	# ------- This code is for 18 -------
	# regResult = regularForm(points, testPts, gam, k)
	# if(regResult != -1):
	# 	numRealRuns += 1
	# 	if(regResult[0] == 0):
	# 		numEin0 += 1

	runs += 1

# print("For K: " + str(k) + " and Gamma: " + str(gam))

# ----------- For Questions 14 and 15 ---------------------
# print("Question 14/15")
# print("Kernel Won: " + str(kernelBetter/float(numRealRuns)))
# print("Regular Won: " + str(regularBetter/float(numRealRuns)))

# ----------- For Question 13 ------------------------
print("Question 13")
print("Kernel Ein: " + str(kernelEin/float(numRuns)))
print("Kernel Eout: " + str(kernelEout/float(numRuns)))
print("Num Non Separable: " + str(numNonSep/float(numRuns)))

# ------------ For Question 16 and 17 -----------------
# print("Question 16/17")
# print("Option A: " + str(optA/float(numRealRuns)))
# print("Option B: " + str(optB/float(numRealRuns)))
# print("Option C: " + str(optC/float(numRealRuns)))
# print("Option D: " + str(optD/float(numRealRuns)))
# print("Option E: " + str(optE/float(numRealRuns)))

# ------------- For Question 18 -----------------------
# print("Question 18")
# print("Num 0 For Regular: " + str(numEin0/float(numRealRuns)))
