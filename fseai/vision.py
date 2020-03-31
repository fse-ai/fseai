"""
Computer Vision Packages
"""
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from fseai import fseai_models as models


def input_mnist():
    """
    Function to load mnist dataset
    """
    trainset = datasets.MNIST('', download=True, train=True,
                              transform=transforms.ToTensor())
    testset = datasets.MNIST('', download=True, train=False,
                             transform=transforms.ToTensor())

    print('MNIST train and test data set Loaded!')
    return trainset, testset


def build_mnist_model():
    """
    Function to build a fully connected model
    """
    # Import the Fully Connected network model from fseai_models
    model = models.FCModel
    input_size = 784
    hidden_size = [128, 64]
    output_size = 10
    model = model(input_size, hidden_size, output_size)
    return model


def train_mnist(model, trainset):
    """
    Function to train mnist data with Fully Connected model
    """
    train_loader = DataLoader(dataset=trainset, batch_size=64, shuffle=True)
    # Loss and optimizer
    lossfunction = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    num_epochs = 10
    loss_mnist = []
    accuracy_mnist = []
    print('Training ......')
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Flatten the input images of [28,28] to [1,784]
            images = images.reshape(-1, 784)

            # Forward Pass
            outputs = model(images)
            loss = lossfunction(outputs, labels)
            loss_mnist.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy_mnist.append(correct/total)

        print('Epoch %s, Training Accuracy: %s' % (epoch, (correct/total) * 100))
    print('Model training with MNIST train data set completed!')
    return model


def test_mnist(model, testset):
    """
    Test the cnn model for mnist test dataset
    """
    test_loader = DataLoader(dataset=testset, batch_size=64, shuffle=False)
    model.eval()
    with torch.no_grad():
        correctly_classified = 0
        total_classified = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 784)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_classified += labels.size(0)
            correctly_classified += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the test dataset: %s'
          % ((correctly_classified/total_classified) * 100))


def visualise_sampledata(samples, dataset):
    """
    Visualise six random samples of dataset with labels
    """
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axes
    a = [ax1, ax2, ax3, ax4, ax5, ax6]
    for i in range(0, 6):
        n = samples[i]
        data = dataset.data[n]
        target = dataset.targets[n]
        a[i].imshow(data, cmap='gray')
        a[i].set_title('Target = %s' % (target.item()))
        fig.tight_layout()


def visualise_test(samples, dataset, model):
    """
    Visualise six random samples of test dataset and corresponding predictions
    """
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axes
    a = [ax1, ax2, ax3, ax4, ax5, ax6]
    for i in range(0, 6):
        n = samples[i]
        data = dataset.data[n]
        image = data.reshape(-1, 784).float()
        target = dataset.targets[n]
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        a[i].imshow(data, cmap='gray')
        a[i].set_title('Target = %s, Prediction = %s' % (target.item())
                       % (predicted.item()))
        fig.tight_layout()
