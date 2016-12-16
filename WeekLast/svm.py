import csv
from sklearn.svm import SVC
import numpy as np
import math

points = [[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]]
yMat = [-1, -1, -1, 1, 1, 1, 1]

transformedPoints = []
for point in points:
	x1 = point[0]
	x2 = point[1]
	z1 = math.pow(x2, 2) - (2 * x1) - 1
	z2 = math.pow(x1, 2) - (2 * x2) + 1
	transformedPoints.append([z1, z2])

# print(transformedPoints)

def checkWeights(w1, w2, b):
	for i in range(len(transformedPoints)):
		point = transformedPoints[i]
		yVal = yMat[i]
		test = np.sign((point[0] * w1) + (point[1] * w2) + b)
		if(test != yVal):
			print(i)
			print "Wrong Option"
			return
	print "Right Option"

# ------- Question 12 -------------
clf = SVC()
clf.set_params(C = float('inf'), kernel = 'poly', coef0 = 1.0, gamma = 1.0, degree = 2)
X = np.array(transformedPoints)
y = np.array(yMat)

clf.fit(X, y)

numSupport = clf.support_vectors_
print(len(numSupport))

# -----Question 11---------
# print("\nOption A:")
# checkWeights(-1, 1, -0.5)
# print("\nOption B:")
# checkWeights(1, -1, -0.5)
# print("\nOption C:")
# checkWeights(1, 0, -0.5)
# print("\nOption D:")
# checkWeights(0, 1, -0.5)
