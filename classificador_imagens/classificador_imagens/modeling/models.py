from loguru import logger
import torch
from torch import nn


class Cifar10CNN(nn.Module):
    """Arquitetura CNN estilo VGG (duplos blocos convolucionais com Batch Normalization e Dropout).

    Projetada para alcançar 80-86% de acurácia no CIFAR-10, com suporte a aceleração na Intel Arc 140V.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Bloco Convolucional 1 (32 filtros)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )

        # Bloco Convolucional 2 (64 filtros)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )

        # Bloco Convolucional 3 (128 filtros)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4),
        )

        # Cabeçalho Denso (Dense Head)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


def create_cifar10_cnn(
    num_classes: int = 10,
    device: torch.device | None = None,
) -> Cifar10CNN:
    """Cria e inicializa o modelo Cifar10CNN no dispositivo indicado."""
    logger.info("Criando modelo CNN estilo VGG (duplos blocos convolucionais)...")
    model = Cifar10CNN(num_classes=num_classes)
    if device is not None:
        model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Modelo criado com sucesso. Total de parâmetros treináveis: {total_params:,}")
    return model
