import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import batchify_data, run_epoch, train_model, Flatten
import utils_multiMNIST as U
path_to_data_dir = '../Datasets/'
use_mini_dataset = True

batch_size = 64
nb_classes = 10
nb_epoch = 30
num_classes = 10
img_rows, img_cols = 42, 28 # input image dimensions



class CNN(nn.Module):

    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        # initialize first set of CONV => RELU => POOL layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                            kernel_size=(3, 3))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        # initialize second set of CONV => RELU => POOL layers
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                            kernel_size=(3, 2))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # flatten
        self.flatten = nn.Flatten()
        # initialize first (and only) set of FC => RELU layers

        self.fc1 = nn.Linear(in_features=3456, out_features=500)
        self.dropout = nn.Dropout(p=0.5)

        # classifier for the 2 outputs
        self.fc2_1 = nn.Linear(in_features=500, out_features=num_classes)
        self.fc2_2 = nn.Linear(in_features=500, out_features=num_classes)
        self.output1 = nn.LogSoftmax(dim=1)
        self.output2 = nn.LogSoftmax(dim=1)


    def forward(self, x):
        # pass the input through our first set of CONV => RELU =>
        # POOL layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        # pass the output from the previous layer through the second
        # set of CONV => RELU => POOL layers
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        # flatten the output from the previous layer and pass it
        # through our only set of FC => RELU layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        # pass the output to our softmax classifier to get our output
        # predictions
        x1 = self.fc2_1(x)
        x2 = self.fc2_2(x)
        out_first_digit = self.output1(x1)
        out_second_digit = self.output2(x2)


        return out_first_digit, out_second_digit

def main(lr=0.01, momentum=0.9, nesterov=False, n_epochs=30, optim="SGD"):
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir, use_mini_dataset)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model
    input_dimension = img_rows * img_cols
    model = CNN(input_dimension) # TODO add proper layers to CNN class above

    # Train
    train_model(train_batches, dev_batches, model,lr, momentum, nesterov, n_epochs, optim)

    ## Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

if __name__ == '__main__':
    # Specify seed for deterministic behavior, then shuffle. Do not change seed for official submissions to edx
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    accuracies = []
    accuracies.append(main())
    batch_size = 34
    accuracies.append(main())
    batch_size = 64
    accuracies.append(main(lr=0.01, momentum=0))
    accuracies.append(main(lr=0.1, momentum=0.9))
    accuracies.append(main(optim="ADAM", lr=0.0001))
    accuracies.append(main(n_epochs=50))
    accuracies.append(main(nesterov=True))
    print(accuracies)

