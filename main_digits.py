from utils import *
from sklearn import neural_network
from neural_network import MyMLPClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter('ignore')


def main():
    # step1: load the data
    inputs, outputs, outputNames = loadDigitsData()

    # step2: split data into train and test
    trainInputs, trainOutputs, testInputs, testOutputs = splitData(inputs, outputs)
    # histogram(trainOutputs, outputNames)

    # step3: normalise data
    scaler = StandardScaler()
    scaler.fit(trainInputs)  # fit only on training data
    trainInputs = scaler.transform(trainInputs)  # apply same transformation to train data
    testInputs = scaler.transform(testInputs)  # apply same transformation to test data

    # step4: training and testing the classifier
    toolClassifier = neural_network.MLPClassifier()
    toolClassifier.fit(trainInputs, trainOutputs)

    predictedLabels = toolClassifier.predict(testInputs)
    acc, prec, recall, cm = evalMultiClass(np.array(testOutputs), predictedLabels, outputNames)

    print("Tool testing:")
    print('\tacc: ', acc)
    print('\tprecision: ', prec)
    print('\trecall: ', recall)
    print()

    myClassifier = MyMLPClassifier(no_iterations=3, hidden_layer_sizes=(10,), learning_rate=0.1, batch_size=30, verbose=True)
    myClassifier.fit(trainInputs, trainOutputs)

    predictedLabels = myClassifier.predict(testInputs)
    acc, prec, recall, cm = evalMultiClass(np.array(testOutputs), predictedLabels, outputNames)

    print("My classifier testing:")
    print('\tacc: ', acc)
    print('\tprecision: ', prec)
    print('\trecall: ', recall)
    print()
    myClassifier.plot_loss()

    # Test from user
    digits = [[]]
    with open("digit", "r") as f:
        for line in f.readlines():
            for el in line.strip().split(' '):
                if el != '':
                    digits[0].append(float(el))

    digits = scaler.transform(digits)
    predicted = toolClassifier.predict(digits)
    print("Prediction for the 'digit' file with tool:", predicted[0])
    predicted = myClassifier.predict(digits)
    print("Prediction for the 'digit' file without tool:", predicted[0])


if __name__ == '__main__':
    main()
