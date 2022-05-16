import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def loadIrisData():
    from sklearn.datasets import load_iris

    data = load_iris()
    inputs = data['data']
    outputs = data['target']
    outputNames = data['target_names']
    featureNames = list(data['feature_names'])
    inputs = [[feat[featureNames.index('sepal length (cm)')],
               feat[featureNames.index('sepal width (cm)')],
               feat[featureNames.index('petal width (cm)')],
               feat[featureNames.index('petal length (cm)')]]
              for feat in inputs]
    return inputs, outputs, outputNames


def loadDigitsData():
    from sklearn.datasets import load_digits

    data = load_digits()
    inputs = data['data']
    outputs = data['target']
    outputNames = data['target_names']
    return inputs, outputs, outputNames


def loadSepiaData():
    import os
    from PIL import Image
    import numpy as np

    trainInputs = []
    trainOutputs = []
    testInputs = []
    testOutputs = []
    outputNames = ["normal", "sepia"]
    for train_normal in os.listdir("data\\databasesepia\\training_set\\normal\\"):
        image = Image.open("data\\databasesepia\\training_set\\normal\\" + train_normal)
        new_image = image.resize((100, 100))
        data = np.asarray(new_image).flatten().tolist()
        trainInputs.append(data)
        trainOutputs.append(0)
    for train_sepia in os.listdir("data\\databasesepia\\training_set\\sepia\\"):
        image = Image.open("data\\databasesepia\\training_set\\sepia\\" + train_sepia)
        new_image = image.resize((100, 100))
        data = np.asarray(new_image).flatten().tolist()
        trainInputs.append(data)
        trainOutputs.append(1)
    for test_normal in os.listdir("data\\databasesepia\\test_set\\normal\\"):
        image = Image.open("data\\databasesepia\\test_set\\normal\\" + test_normal)
        new_image = image.resize((100, 100))
        data = np.asarray(new_image).flatten().tolist()
        testInputs.append(data)
        testOutputs.append(0)
    for test_sepia in os.listdir("data\\databasesepia\\test_set\\sepia\\"):
        image = Image.open("data\\databasesepia\\test_set\\sepia\\" + test_sepia)
        new_image = image.resize((100, 100))
        data = np.asarray(new_image).flatten().tolist()
        testInputs.append(data)
        testOutputs.append(1)

    return trainInputs, trainOutputs, testInputs, testOutputs, outputNames


def splitData(inputs, outputs):
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    testSample = [i for i in indexes if i not in trainSample]

    trainInputs = [list(inputs[i]) for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [list(inputs[i]) for i in testSample]
    testOutputs = [outputs[i] for i in testSample]

    return trainInputs, trainOutputs, testInputs, testOutputs


def normalisation(trainData, testData):
    scaler = StandardScaler()
    if not isinstance(trainData[0], list):
        # encode each sample into a list
        trainData = [[d] for d in trainData]
        testData = [[d] for d in testData]

        scaler.fit(trainData)  # fit only on training data
        normalisedTrainData = scaler.transform(trainData)  # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data

        # decode from list to raw values
        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
    else:
        scaler.fit(trainData)  # fit only on training data
        normalisedTrainData = scaler.transform(trainData)  # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data
    return normalisedTrainData, normalisedTestData


def histogram(trainOutputs, outputNames):
    plt.hist(trainOutputs, len(outputNames), rwidth=0.8)
    plt.xticks(np.arange(len(outputNames)), outputNames)
    plt.show()


def data2FeaturesMoreClasses(inputs, outputs, outputNames):
    labels = set(outputs)
    noData = len(inputs)
    for crtLabel in labels:
        x = [inputs[i][0] for i in range(noData) if outputs[i] == crtLabel]
        y = [inputs[i][1] for i in range(noData) if outputs[i] == crtLabel]
        plt.scatter(x, y, label=outputNames[crtLabel])
    plt.xlabel('feat1')
    plt.ylabel('feat2')
    plt.legend()
    plt.show()


def evalMultiClass(realLabels, computedLabels, labelNames):
    from sklearn.metrics import confusion_matrix

    confMatrix = confusion_matrix(realLabels, computedLabels)
    acc = sum([confMatrix[i][i] for i in range(len(labelNames))]) / len(realLabels)
    precision = {}
    recall = {}
    for i in range(len(labelNames)):
        precision[labelNames[i]] = confMatrix[i][i] / sum([confMatrix[j][i] for j in range(len(labelNames))])
        recall[labelNames[i]] = confMatrix[i][i] / sum([confMatrix[i][j] for j in range(len(labelNames))])
    return acc, precision, recall, confMatrix
