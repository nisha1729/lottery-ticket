import torch
import torchvision
import torchvision.transforms as transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load CIFAR-10 Dataset
norm_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar_dataset = torchvision.datasets.CIFAR10(root='datasets/', train=True, transform=norm_transform, download=False)
test_dataset = torchvision.datasets.CIFAR10(root='datasets/', train=False, transform=test_transform)


def data_base():
    num_training = 49000
    num_validation = 1000
    batch_size = 200  # 200
    # Prepare the training and validation splits
    dataset_mask = list(range(num_training))
    train_dataset = torch.utils.data.Subset(cifar_dataset, dataset_mask)
    dataset_mask = list(range(num_training, num_training + num_validation))
    val_dataset = torch.utils.data.Subset(cifar_dataset, dataset_mask)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
