from pathlib import Path

from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import typer

from classificador_imagens.config import RAW_DATA_DIR

app = typer.Typer()


def get_dataloaders(
    batch_size: int = 64,
    val_split: float = 0.2,
    data_dir: Path = RAW_DATA_DIR,
    download: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Retorna os DataLoaders (treino, validação e teste) para o CIFAR-10."""
    logger.info("Preparando DataLoaders do dataset CIFAR-10...")

    # Data augmentation leve para o treino
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=2, padding_mode="edge"),
            transforms.ToTensor(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    full_train_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=download,
        transform=train_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=download,
        transform=test_transform,
    )

    # Divisão Treino / Validação reproduzível
    total_train = len(full_train_dataset)
    val_size = int(total_train * val_split)
    train_size = total_train - val_size

    train_subset, val_subset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    logger.info(
        f"Dataset preparado: Treino={train_size} | Validação={val_size} | Teste={len(test_dataset)}"
    )
    return train_loader, val_loader, test_loader


def load_cifar10_numpy(
    data_dir: Path = RAW_DATA_DIR,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Carrega dados em formato numpy (N, 32, 32, 3) normalizados em [0, 1] para plotagens."""
    train_dataset = datasets.CIFAR10(root=str(data_dir), train=True, download=False)
    test_dataset = datasets.CIFAR10(root=str(data_dir), train=False, download=False)

    x_train = train_dataset.data.astype("float32") / 255.0
    y_train = np.array(train_dataset.targets)
    x_test = test_dataset.data.astype("float32") / 255.0
    y_test = np.array(test_dataset.targets)

    return (x_train, y_train), (x_test, y_test)


# Alias para compatibilidade
load_cifar10 = load_cifar10_numpy


@app.command()
def main(
    data_dir: Path = RAW_DATA_DIR,
    download: bool = True,
):
    """Comando CLI para pré-carregar o dataset CIFAR-10."""
    logger.info("Baixando/verificando o dataset CIFAR-10...")
    get_dataloaders(data_dir=data_dir, download=download)
    logger.success("Dataset CIFAR-10 pronto para uso!")


if __name__ == "__main__":
    app()
