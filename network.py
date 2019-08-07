import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from inferno.extensions.layers.reshape import Flatten
from builtins import range
from tqdm import tqdm
import dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 3
num_classes = 10
fc_size = 256
num_epochs = 20  # 50
batch_size = 200    # 200
learning_rate = 0.0002   # 2e-3
learning_rate_decay = 0.0001   # 0.95
reg = 0.001
norm_layer = None


class ConvNet(nn.Module):
    def __init__(self, mask):
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


def train(model):
    max_val_acc = 0
    iter_num = 0
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

        val_acc = validate(model)

        # Saving best model
        if (val_acc > max_val_acc):
            # print("Saving the model...")
            torch.save(model.state_dict(), 'model_early.ckpt')
            max_val_acc = val_acc
            iter_num = i + 1 + epoch*num_training/batch_size
    return iter_num


def validate(model):
    model.eval()

    # Checking validation accuracy
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
    return val_acc


def test(model):
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

