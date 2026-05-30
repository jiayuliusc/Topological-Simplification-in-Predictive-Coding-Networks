"""Dataset loading helpers for PCN experiments."""

from __future__ import annotations

try:
    from .pcn_backend import CONFIG, datasets, transforms
except ImportError:
    from pcn_backend import CONFIG, datasets, transforms


def load_dataset(name="MNIST", root=None):
    """
    Returns (train_dataset, test_dataset, input_dim, output_dim).
    MNIST/FashionMNIST are flattened to vectors. CIFAR10 is also flattened.
    """
    root = str(root or CONFIG.data_dir)
    name_u = name.upper()

    if name_u in ("MNIST", "FASHIONMNIST"):
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda t: t.view(-1))  # flatten 28*28 → 784
        ])
        if name_u == "MNIST":
            train_ds = datasets.MNIST(root=root, train=True, transform=tfm, download=True)
            test_ds  = datasets.MNIST(root=root, train=False, transform=tfm, download=True)
        else:
            train_ds = datasets.FashionMNIST(root=root, train=True, transform=tfm, download=True)
            test_ds  = datasets.FashionMNIST(root=root, train=False, transform=tfm, download=True)
        return train_ds, test_ds, 28*28, 10

    if name_u == "CIFAR10":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda t: t.view(-1))  # flatten 32*32*3 → 3072
        ])
        train_ds = datasets.CIFAR10(root=root, train=True, transform=tfm, download=True)
        test_ds  = datasets.CIFAR10(root=root, train=False, transform=tfm, download=True)
        return train_ds, test_ds, 32*32*3, 10

    raise ValueError(f"Unsupported dataset: {name}")
