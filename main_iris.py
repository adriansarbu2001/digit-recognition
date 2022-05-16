from utils import *
from sklearn import neural_network
from neural_network import *
import warnings
warnings.simplefilter('ignore')


def main():
    # step1: load the data
    inputs, outputs, outputNames = loadIrisData()

    # step2: split data into train and test
    trainInputs, trainOutputs, testInputs, testOutputs = splitData(inputs, outputs)
    # histogram(trainOutputs, outputNames)
    # data2FeaturesMoreClasses(trainInputs, trainOutputs, outputNames)

    # step3: normalise data
    trainInputs, testInputs = normalisation(trainInputs, testInputs)
    # data2FeaturesMoreClasses(trainInputs, trainOutputs, outputNames)

    # step4: training the classifier
    toolClassifier = neural_network.MLPClassifier()
    toolClassifier.fit(trainInputs, trainOutputs)

    myClassifier = MyMLPClassifier(no_iterations=200, hidden_layer_sizes=(5,), learning_rate=0.05, batch_size=10)
    myClassifier.fit(trainInputs, trainOutputs)

    # step5: testing (predict the labels for new inputs_row)

    predictedLabels = toolClassifier.predict(testInputs)
    acc, prec, recall, cm = evalMultiClass(np.array(testOutputs), predictedLabels, outputNames)

    print("Tool testing:")
    print('\tacc: ', acc)
    print('\tprecision: ', prec)
    print('\trecall: ', recall)

    predictedLabels = myClassifier.predict(testInputs)
    acc, prec, recall, cm = evalMultiClass(np.array(testOutputs), predictedLabels, outputNames)

    print("My classifier testing:")
    print('\tacc: ', acc)
    print('\tprecision: ', prec)
    print('\trecall: ', recall)

    myClassifier.plot_loss()


if __name__ == '__main__':
    main()
