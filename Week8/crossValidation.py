import csv
import numpy as np
import sklearn.svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

# These stand for how the data should be classified.
firstVal = 1.0
secondVal = 5.0
# These are all of the possible values for C.
Coptions = [0.0001, 0.001, 0.01, 0.1, 1]
# This holds the Ecv for each value of C.
Ecv = [0, 0, 0, 0, 0]
# This holds the final count for each value of C. The count is the
# number of times each C gave the lowest Ecv for the run.
Ccount = [0, 0, 0, 0, 0]
numRuns = 0

while(numRuns < 100):

	xData = []
	yData = []
	# This classifies the data using firstVal and secondVal.
	with open('features.train.csv', 'rb') as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			x = row[1:]
			if(float(row[0]) == firstVal):
				yData.append(1)
				xData.append(x)
			elif(float(row[0]) == secondVal):
				yData.append(-1)
				xData.append(x)


	# This performs the partition of the data set.
	cv = ShuffleSplit(n_splits=10, test_size=0.1, train_size=0.9)
	curE = []

	# For every possible C, this computes the score for the partitioned data.
	for i in range(len(Coptions)):
		clf = SVC(kernel = 'poly', C=Coptions[i], degree=2, coef0=1.0, gamma=1)
		scores = cross_val_score(clf, xData, yData, cv=cv)
		# We get the '1 - scores' instead of just the scores because we want the
		# error not the accuracy.
		scores = 1 - scores
		# This appends the mean of the error to the list for the run.
		curE.append(np.mean(scores))
		# This adds the mean of the error to the overall accumulation
		Ecv[i] += np.mean(scores)

	# This chooses the best C based on which one has the lowest Ecv.
	Ccount[curE.index(min(curE))] += 1
	numRuns += 1

# This prints out the Ecv for each C value.
for i in Ecv:
	print(i/100)
# This prints out the count for each C value.
print(Ccount)
