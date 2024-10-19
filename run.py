# Import necessary libraries

# Import the PyTorch library
# PyTorch is an open-source machine learning library used for applications such as computer vision & natural language processing
import torch

# Import the neural network module from PyTorch
# torch.nn provides classes & functions to build neural networks
import torch.nn as nn

# Import the optimization module from PyTorch
# torch.optim provides various optimization algorithms such as SGD, Adam, etc.
import torch.optim as optim

# Import the torchvision library
# torchvision provides tools to handle image & video data
import torchvision

# Import the transforms module from torchvision
# torchvision.transforms provides common image transformations for data augmentation & preprocessing
import torchvision.transforms as transforms

# Import the matplotlib library for plotting
# matplotlib.pyplot provides functions to create various types of plots & visualizations
import matplotlib.pyplot as plt

# Import the tqdm library for progress bars
# tqdm provides a fast, extensible progress bar for loops & tasks
from tqdm import tqdm

# Import the numpy library
# numpy provides support for large, multi-dimensional arrays & matrices, along with a collection of mathematical functions to operate on these arrays
import numpy as np

from utilities import *

# Device configuration (use GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#@title Track accuracy
def track_accuracy(model, loader):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        correct = 0
        total = 0
        for images, labels in loader:
            images = images.to(device)  # Move images to the configured device
            labels = labels.to(device)  # Move labels to the configured device
            outputs = model(images)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
            total += labels.size(0)  # Increment total by the number of labels
            correct += (predicted == labels).sum().item()  # Increment correct by the number of correct predictions
        accuracy = 100 * correct / total  # Calculate accuracy
    model.train()  # Set the model back to training mode
    return accuracy

#@title Training
def training(model, num_epochs, train_loader, criterion, optimizer,test_loader):
    # Train the model
    total_step = len(train_loader)  # Total number of batches
    loss_list = []  # List to store loss values
    acc_train = []  # List to store training accuracy values
    acc_test = []  # List to store testing accuracy values

    # Loop over the number of epochs
    for epoch in range(num_epochs):
        # Loop over each batch in the training DataLoader

        pbar = tqdm(train_loader,
                    desc = f"Training: {epoch + 1:03d}/{num_epochs}",
                    ncols = 125,
                    leave = True)


        # Creating running loss & accuracy empty lists
        running_loss = []
        running_accuracy = []

        for i, (images, labels) in enumerate(pbar, 1):
            # Important note: In PyTorch, the images in a batch is typically represented as (batch_size, channels, height, width)
            # For example, (100, 1, 28, 28) for the MNIST data
            images = images.to(device)  # Move images to the configured device
            labels = labels.to(device)  # Move labels to the configured device

            # Forward pass: compute predicted y by passing x to the model
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            running_loss.append(loss.item())

            # Backward pass & optimize
            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            # Add the runnign loss
            pbar.set_postfix(loss = f"{sum(running_loss)/i : 10.6f}")

        # Get the average of all losses in the running_loss as the loss of the current epoch
        loss_list.append(sum(running_loss)/i)

        # Track accuracy on the train set
        acc_train.append(track_accuracy(model, train_loader))

        # Track accuracy on the test set
        acc_test.append(track_accuracy(model, test_loader))

    return model, loss_list, acc_train, acc_test

def main():
    #@title Main run

    args = myargs()



    # Hyperparameters
    num_epochs = args.num_epochs  # Number of times the entire dataset is passed through the model
    batch_size = args.batch_size  # Number of samples per batch to be passed through the model
    learning_rate = args.learning_rate  # Step size for parameter updates

    # MNIST dataset
    # Download & load the training dataset, applying a transformation to convert images to PyTorch tensors
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=True,
                                            transform=transforms.ToTensor(),
                                            download=True)

    # Download & load the test dataset, applying a transformation to convert images to PyTorch tensors
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=False,
                                            transform=transforms.ToTensor())

    # Data loader
    # DataLoader provides an iterable over the dataset with support for batching, shuffling, & parallel data loading
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

    # DataLoader for the test dataset, used for evaluating the model
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)

    # Create an instance of the model & move it to the configured device (GPU/CPU)
    model = ConvNet().to(device)

    # Print the model for observation
    # You can use the following libraries to observe the flowchart of the models similar to TensorFlow
    # from torchview import draw_graph; from torchviz import make_dot
    # see more at:
    # https://github.com/mert-kurttutan/torchview
    # https://github.com/szagoruyko/pytorchviz
    print(model)

    # Loss & optimizer
    # CrossEntropyLoss combines nn.LogSoftmax & nn.NLLLoss in one single class
    criterion = nn.CrossEntropyLoss()
    # Adam optimizer with the specified learning rate
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training the model
    model, loss_list, acc_train, acc_test = training(model, num_epochs,
                                                    train_loader, criterion, optimizer,test_loader)
    # Make some plots!
    plot(loss_list, acc_train, acc_test)


if __name__ == "__main__":
    main()
