import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from fseai import fseai_models as models


def input_dataset(name):
    """Function to load specified dataset from torchvision.datasets

    Args:
        name::str
            The name of the dataset in PyTorch to be imported
            (MNIST, FashionMMNIST, CIFAR10, KMNIST, QMNIST)


    Returns:
        trainset::torchvision.datasets
            training data from dataset
        testset::torchvision.datasets
            testing data from dataset
    """
    name = name.replace(" ", "").upper()
    concatname = 'datasets' + '.' + name
    function_map = {'datasets.MNIST': datasets.MNIST, 'datasets.FASHIONMNIST': datasets.FashionMNIST,
                    'datasets.CIFAR10': datasets.CIFAR10, 'datasets.KMNIST': datasets.KMNIST,
                    'datasets.QMNIST': datasets.QMNIST}
    selection = function_map[concatname]
    trainset = selection('.', download=True, train=True, transform=transforms.ToTensor())
    testset = selection('.', download=True, train=False, transform=transforms.ToTensor())

    print('Selected train and test data set Loaded!')
    return trainset, testset


def build_model(trainset):
    """Builds a Fully connected model from the fseai.models by finding the
    flattened image size and number of output classes from training dataset


    Args:
        trainset::torchvision.datasets
            Provide training set to find features from images in the dataset

    Returns:
        model::fseai_models.FCModel
            Fully connected model with two hidden layers
    """
    image_size = trainset.data[1].shape
    if len(image_size) == 2:
        input_size = image_size[0] * image_size[1]

    if len(image_size) == 3:
        input_size = image_size[0] * image_size[1] * image_size[2]

    if isinstance(trainset.targets, torch.Tensor):
        output_size = (max(trainset.targets).item() + 1)
    else:
        output_size = max(trainset.targets) + 1

    # Import the Fully Connected network model from fseai_models
    model = models.FCModel
    hidden_size = [128, 64]
    model = model(input_size, hidden_size, output_size)
    return model


def train_model(model, trainset):
    """Function to train mnist data with Fully Connected model


    Args:
        model::fseai_models.FCModel
            The fully connected model build for the dataset to be trained
        trainset::torchvision.datasets
            The training dataset to train the FC model on

    Returns:
        model::fseai_models.FCModel
            The FC model after training
    """
    train_loader = DataLoader(dataset=trainset, batch_size=64, shuffle=True)
    # Loss and optimizer
    lossfunction = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    num_epochs = 6
    loss_mnist = []
    accuracy_mnist = []
    print('Training ......')
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Flatten the input images of [28,28] to [1,784]
            images = images.reshape(-1, model.layer1.in_features)

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
            accuracy_mnist.append(correct / total)

        print('Epoch %s, Training Accuracy: %s' % (epoch, (correct / total) * 100))
    print('Model training with MNIST train data set completed!')
    return model


def test_model(model, testset):
    """Test the model for mnist test dataset


    Args:
        model::fseai_models.FCModel
            The fully connected model trained to be tested
        testset::torchvision.datasets
            The testing dataset to validate the training of FC model

    Returns:
        test_accuracy::float
            The accuracy of the performance of the model on testing dataset
    """
    input_size = model.layer1.in_features
    test_loader = DataLoader(dataset=testset, batch_size=64, shuffle=False)
    model.eval()
    with torch.no_grad():
        correctly_classified = 0
        total_classified = 0
        for images, labels in test_loader:
            images = images.reshape(-1, input_size)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_classified += labels.size(0)
            correctly_classified += (predicted == labels).sum().item()

    test_accuracy = (correctly_classified / total_classified) * 100
    print('Test Accuracy of the model on the test dataset: %s' % (test_accuracy))
    return test_accuracy


def visualise_sampledata(sample, dataset):
    """Visualise any random sample of dataset with labels


    Args:
        sample::int
            Random sample number from the trainset
        dataset::torchvision.datasets
            The training dataset display data from
    """
    n = sample
    data = dataset.data[n]
    target = dataset.targets[n]
    if isinstance(target, torch.Tensor):
        target = target
    else:
        target = torch.Tensor([target])
    plt.imshow(data, cmap='gray')
    plt.title('Target = %s' % (target.item()))


def visualise_sampledata_set(samples, dataset):
    """Visualise six random samples of dataset with labels


    Args:
        samples::int
            6 random sample number from the trainset
        dataset::torchvision.datasets
            The training dataset display data from
    """
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axes
    a = [ax1, ax2, ax3, ax4, ax5, ax6]
    for i in range(0, 6):
        n = samples[i]
        data = dataset.data[n]
        target = dataset.targets[n]
        if isinstance(target, torch.Tensor):
            target = target
        else:
            target = torch.Tensor([target])
        a[i].imshow(data, cmap='gray')
        a[i].set_title('Target = %s' % (target.item()))
        fig.tight_layout()


def visualise_test(sample, dataset, model):
    """Visualise a random samples of test dataset and corresponding predictions

    Args:
        sample::int
            Random sample number from the trainset
        dataset::torchvision.datasets
            The training dataset display data from
        model::fseai_models.FCModel
            The fully connected model trained to be tested
    """
    n = sample
    data = dataset.data[n]
    input_size = model.layer1.in_features
    if isinstance(data.reshape(-1, input_size), torch.Tensor):
        image = data.reshape(-1, input_size).float()
        target = dataset.targets[n]
    else:
        # if type(testset.data.reshape(-1, input_size)) == 'numpy.ndarray':
        image_numpy = data.reshape(-1, input_size)
        image = torch.from_numpy(image_numpy).float()
        target_numpy = dataset.targets[n]
        target = torch.Tensor([target_numpy])

    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    plt.imshow(data, cmap='gray')
    plt.title('Target = %s, Prediction = %s' % (target.item(), predicted.item()))


def visualise_test_set(samples, dataset, model):
    """Visualise six random samples of test dataset and corresponding predictions

    Args:
        samples::int
            6 random random samples number from the trainset
        dataset::torchvision.datasets
            The training dataset display data from
        model::fseai_models.FCModel
            The fully connected model trained to be tested
    """
    input_size = model.layer1.in_features
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axes
    a = [ax1, ax2, ax3, ax4, ax5, ax6]
    for i in range(0, 6):
        n = samples[i]
        data = dataset.data[n]
        if isinstance(data.reshape(-1, input_size), torch.Tensor):
            image = data.reshape(-1, input_size).float()
            target = dataset.targets[n]
        else:
            # if type(testset.data.reshape(-1, input_size)) == 'numpy.ndarray':
            image_numpy = data.reshape(-1, input_size)
            image = torch.from_numpy(image_numpy).float()
            target_numpy = dataset.targets[n]
            target = torch.Tensor([target_numpy])

        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        a[i].imshow(data, cmap='gray')
        a[i].set_title('Target = %s, Prediction = %s' % (target.item(), predicted.item()))
        fig.tight_layout()
