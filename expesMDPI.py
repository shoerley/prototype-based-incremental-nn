import pandas as pd
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import INN.ci_classifieur as INN

column_names = ["accuracy", "rhoProto", "train_size", "test_size", "train_pc", "classifierName"]


def getData():
	irisDataset = load_iris()
	X = irisDataset.data
	y = irisDataset.target

	return X, y


def getClassifiers():
	return [
		KNeighborsClassifier(3),
		SVC(kernel="linear", C=0.025),
		SVC(gamma=2, C=1),
		GaussianProcessClassifier(1.0 * RBF(1.0)),
		DecisionTreeClassifier(max_depth=5),
		RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
		MLPClassifier(alpha=1, max_iter=1000),
		AdaBoostClassifier(),
		GaussianNB(),
		QuadraticDiscriminantAnalysis(),
		INN.ClassifieurIncremental(s_inf=1, s_conf=0.1, alpha=0.01, verbose=False)
	], [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
	"INN"
	]


def classifyMultiple():
	X, y = getData()
	results = pd.DataFrame(columns=column_names)

	# for each trainset increased by 1% from size 5% to 95%
	for i in range(5, 95):
		X_train, X_test, y_train, y_test = train_test_split(
			X, y, train_size=0.01 * i, random_state=1, stratify=y)

		# scale data
		scaler = StandardScaler()
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)

		classifiers, classifierNames = getClassifiers()
		accuracyList = []
		rhoProtoList = []
		trainSizeList = []
		testSizeList = []
		trainPercentageList = []
		classifierNameList = []

		# fit and score each classifier on the train set
		for clf, name in zip(classifiers, classifierNames):
			clf.fit(X_train, y_train)
			acc = clf.score(X_test, y_test)
			accuracyList.append(acc)
			trainSizeList.append(X_train.shape[0])
			testSizeList.append(X_test.shape[0])
			trainPercentageList.append(X_train.shape[0] / X.shape[0])
			classifierNameList.append(name)
			if hasattr(clf, "p_"):
				rhoProtoList.append(clf.p_ / X_train.shape[0])
			else:
				rhoProtoList.append(0)



		data = {col: lst for col, lst in
				zip(column_names, [accuracyList, rhoProtoList, trainSizeList, testSizeList, trainPercentageList, classifierNameList])}
		df = pd.DataFrame(data)
		results = pd.concat([results, df])


	# will not overwrite
	# results.to_csv("resultsMDPI.csv", index=False)
	# results.to_excel("resultsMDPI.xlsx", index=False)

	print(results.head())

def makePlots():
	results = pd.read_csv("resultsMDPI.csv")

	_, classifierNames = getClassifiers()

	x = results.loc[results['classifierName'] == 'INN', 'train_pc'] * 100
	fig = plt.figure()
	plt.xlabel("Train set size (in %)")
	plt.ylabel("Accuracy")

	for name in classifierNames:
		y = results.loc[results['classifierName'] == name, 'accuracy']
		if name == "INN":
			plt.plot(x, y, label=name, color="black")
		else:
			plt.plot(x, y, label=name, linestyle="-.")


	rhoProto = results.loc[results['classifierName'] == 'INN', 'rhoProto']
	plt.plot(x, rhoProto, label="Number of prototypes (INN)", linestyle=":")



	plt.legend()
	plt.show()

makePlots()
