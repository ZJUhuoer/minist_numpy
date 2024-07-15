from CNN.network import *
from CNN.utils import *

from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Train a convolutional neural network.')

if __name__ == '__main__':

    args = parser.parse_args()

    # Train the network and get the parameters
    params, cost = train()

    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    # Get test data
    m = 10000
    X = extract_data('./MNIST/raw/t10k-images-idx3-ubyte')
    y_dash = extract_labels('./MNIST/raw/t10k-labels-idx1-ubyte').reshape(m, 1)

    # Normalize the data
    X -= int(np.mean(X))  # subtract mean
    X /= int(np.std(X))   # divide by standard deviation
    test_data = np.hstack((X, y_dash))

    X = test_data[:, 0:-1]
    X = X.reshape(len(test_data), 1, 28, 28)
    y = test_data[:, -1]

    corr = 0
    digit_count = [0 for i in range(10)]
    digit_correct = [0 for i in range(10)]

    print("\nComputing accuracy over test set:")
    t = tqdm(range(len(X)), leave=True)

    for i in t:
        x = X[i]
        pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
        digit_count[int(y[i])] += 1
        if pred == y[i]:
            corr += 1
            digit_correct[pred] += 1

        t.set_description("Acc:%0.2f%%" % (float(corr / (i + 1)) * 100))

    print("Overall Accuracy: %.2f%%" % (float(corr / len(test_data) * 100)))