import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from inferno.extensions.layers.reshape import Flatten
from builtins import range
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from pylab import *

# --------------------------------
# Device configuration
# --------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)

# --------------------------------
# Hyper-parameters
# --------------------------------
input_size = 3
num_classes = 10
fc_size = 256
num_epochs = 10  # 50
batch_size = 200    # 200
learning_rate = 0.0002   # 2e-3
learning_rate_decay = 0.0001   # 0.95
reg = 0.001
num_training = 49000
num_validation = 1000
norm_layer = None
prune_percent = 20
prune_iter = 20
validation_split = .02      # Percentage (*100) of data to be put into validation set
data_split = .1             # Percentage (*100) of data to be split into two sets

# -------------------------------------------------
# Load the CIFAR-10 dataset
# -------------------------------------------------

norm_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/', train=True, transform=norm_transform, download=False)
test_dataset = torchvision.datasets.CIFAR10(root='datasets/', train=False, transform=test_transform)

# -------------------------------------------------
# Prepare the training and validation splits
# -------------------------------------------------

random_seed = 42
shuffle_dataset = True
dataset_size = len(cifar_dataset)                           # Validation and Training Set
test_dataset_size = len(test_dataset)                       # Test set
indices = list(range(dataset_size))                         # Indices for Validation and train set
test_indices = list(range(test_dataset_size))               # Indices for test set
d_split = int(np.floor(data_split * dataset_size))          # Splitting index for separating two datasets among train and val set
t_split = int(np.floor(data_split * test_dataset_size))     # Splitting index for separating two datasets among test set
split = int(np.floor(validation_split * dataset_size))      # Splitting index for splitting into train and val
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    np.random.shuffle(test_indices)
data2, data1 = indices[d_split:], indices[:d_split]             # Splitting val and train data into two sets of val and train
train_indices1, val_indices1 = data1[split:], data1[:split]     # Splitting into val and train for first set
train_indices2, val_indices2 = data2[split:], data2[:split]     # Splitting into val and train for second test
test_indices2, test_indices1 = test_indices[t_split:], test_indices[:t_split]       # Splitting test set into two sets

# Creating PT data samplers and loaders:
train_sampler1 = SubsetRandomSampler(train_indices1)
valid_sampler1 = SubsetRandomSampler(val_indices1)
test_sampler1 = SubsetRandomSampler(test_indices1)
train_sampler2 = SubsetRandomSampler(train_indices2)
valid_sampler2 = SubsetRandomSampler(val_indices2)
test_sampler2 = SubsetRandomSampler(test_indices2)

# -------------------------------------------------
# Data loader
# -------------------------------------------------

train_loader1 = torch.utils.data.DataLoader(cifar_dataset, batch_size=batch_size, sampler=train_sampler1)
val_loader1 = torch.utils.data.DataLoader(cifar_dataset, batch_size=batch_size, sampler=valid_sampler1)
test_loader1 = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler1)
train_loader2 = torch.utils.data.DataLoader(cifar_dataset, batch_size=batch_size, sampler=train_sampler2)
val_loader2 = torch.utils.data.DataLoader(cifar_dataset, batch_size=batch_size, sampler=valid_sampler2)
test_loader2 = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler2)

# -------------------------------------------------
# Convolutional neural network
# -------------------------------------------------

class ConvNet(nn.Module):
    def __init__(self, input_size, num_classes, mask):
        super(ConvNet, self).__init__()
        layers = []
        layers.append(nn.Conv2d(input_size, 64, kernel_size=3))
        layers.append(nn.Conv2d(64, 64, kernel_size=3))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        layers.append(Flatten())
        layers.append(nn.Linear(64*13*13, fc_size))
        layers.append(nn.Linear(fc_size, fc_size))
        layers.append(nn.Linear(fc_size, num_classes))
        self.layers = nn.Sequential(*layers)

        self.mask = mask

    def forward(self, x):
        if self.mask:
            for name, param in self.named_parameters():
                if 'weight' in name:
                    param.data = param.data * self.mask[name].float()
        out = self.layers(x)
        return out

# -------------------------------------------------
# Function Definitions
# -------------------------------------------------


def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_initial_mask(model):
    # Initialising mask with all ones
    mask = dict()
    for name, param in model.named_parameters():
        mask[name] = (torch.ones(param.size()).byte().to(device))
    return mask


def train(model, train_loader, val_loader):
    max_val_acc = 0
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)
    model.train()
    for epoch in tqdm(range(num_epochs)):
        for i, (images, labels) in enumerate(train_loader):

            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # TODO: Learning rate decay
        # Code to update the lr
        # lr *= learning_rate_decay
        # update_lr(optimizer, lr)

        val_acc = validate(model, val_loader)

        # Saving best model
        if (val_acc > max_val_acc):
            # print("Saving the model...")
            torch.save(model.state_dict(), 'model_early.ckpt')
            max_val_acc = val_acc
            iter_num = i + 1 + epoch*num_training/batch_size
    return iter_num


def validate(model, val_data):
    model.eval()

    # Checking validation accuracy
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_data:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
    return val_acc


def test(model, test_loader):
    # TESTING
    model.eval()

    # Load the best model
    best_model = torch.load("model_early.ckpt")
    model.load_state_dict(best_model)

    # Calculating test accuracy
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if total == 1000:
                break

    return 100*correct/total


def prune(prune_percent, model, mask):
    mask_dict = dict()
    for name, param in model.named_parameters():
        if 'weight' in name and '6' not in name:
            sort_array = torch.sort(torch.abs(torch.masked_select(param, mask[name])))[0]
            thres_index = int(prune_percent * len(sort_array) / 100)
            threshold = sort_array[thres_index]
            mask_dict[name] = torch.where(torch.abs(param).cpu() <= threshold.cpu(),
                                   torch.zeros(mask[name].shape).byte().cpu(),
                                   mask[name].cpu()).byte().cuda()
        if '6.weight' in name:
            sort_array = torch.sort(torch.abs(torch.masked_select(param, mask[name])))[0]
            thres_index = int(prune_percent * len(sort_array) / 200)
            threshold = sort_array[thres_index]
            mask_dict[name] = torch.where(torch.abs(param).cpu() <= threshold.cpu(),
                                   torch.zeros(mask[name].shape).byte().cpu(),
                                   mask[name].cpu()).byte().cuda()
    return mask_dict


def get_weights_remaining(mask_dict):
    non_zeros = 0
    total = 0

    for name, mask in mask_dict.items():
        non_zeros += torch.sum(mask == 1).item()
        total += mask.numel()

    return 100*non_zeros/total


# -------------------------------------------------
# Main
# -------------------------------------------------


if __name__ == "__main__":
    percent_weights_remaining = []
    test_accuracies_samedata = []
    test_accuracies_newdata = []
    iter_history = []

    # Create CNN model
    model = ConvNet(input_size, num_classes, mask=None).to(device)
    model.apply(weights_init)
    torch.save(model.state_dict(), 'model_initial.ckpt')

    # Train and test on <data_split> % of the dataset
    iter_history.append(train(model, train_loader1, val_loader1))
    test_accuracies_samedata.append(test(model, test_loader1))
    test_accuracies_newdata.append(test(model, test_loader1))
    initial_mask = get_initial_mask(model)
    percent_weights_remaining.append(get_weights_remaining(initial_mask))
    new_mask = prune(prune_percent, model, initial_mask)

    # For comparison, prune and train on remaining data
    for i in range(1, prune_iter):
        try:
            initial_model = torch.load("model_initial.ckpt")
            model.load_state_dict(initial_model)
            model.mask = new_mask

            # Train on full dataset
            iter_history.append(train(model, train_loader2, val_loader2))
            test_accuracies_newdata.append(test(model, test_loader2))
            percent_weights_remaining.append(get_weights_remaining(new_mask))

            # Reset, train on subset
            initial_model = torch.load("model_initial.ckpt")
            model.load_state_dict(initial_model)
            model.mask = new_mask
            train(model, train_loader1, val_loader1)
            test_accuracies_samedata.append(test(model, test_loader1))

            # Prune on subset iteratively
            new_mask = prune(prune_percent, model, new_mask)

        except IndexError:
            break

    print("Accuracies on 90% dataset")
    print(test_accuracies_newdata)
    print("Accuracies on 10% dataset")
    print(test_accuracies_samedata)
    print("Number of Iterations")
    print(iter_history)
    #
    # f1 = plt.figure(1)
    # plt.plot(percent_weights_remaining[:len(test_accuracies_newdata)].reverse(), test_accuracies_newdata.reverse(), 'r*--', label= "Accuracy on 90% dataset")
    # plt.plot(percent_weights_remaining[:len(test_accuracies_samedata)].reverse(), test_accuracies_samedata.reverse(), 'o--', label= "Accuracy on 10% dataset")
    # plt.xlabel("Percentage of Weights Remaining")
    # plt.ylabel("Early Stopping Test Accuracy")
    # plt.title('Winning ticket from %d percent of dataset' % (data_split*100))
    # plt.grid()
    # plt.legend()
    # # f1.gca().invert_xaxis()
    # # plt.savefig("prune_acc_split_iter_num.png")
    #
    #
    # f2 = plt.figure(2)
    # plt.plot(percent_weights_remaining[:len(test_accuracies_newdata)].reverse(), iter_history.reverse(), 'b*--')
    # plt.xlabel("Percentage of Weights Remaining")
    # plt.ylabel("Early Stopping Iteration")
    # plt.title('Winning ticket from %d percent of dataset' % (data_split*100))
    # plt.grid()
    # # f2.gca().invert_xaxis()
    # plt.savefig("prune_acc_split_iter.png")
