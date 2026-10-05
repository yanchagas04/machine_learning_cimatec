import copy
from pathlib import Path

from loguru import logger
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import typer

from classificador_imagens.config import DEFAULT_MODEL_PATH, get_device
from classificador_imagens.dataset import get_dataloaders
from classificador_imagens.modeling.models import create_cifar10_cnn

app = typer.Typer()


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    learning_rate: float = 0.001,
    device: torch.device | None = None,
    patience_lr: int = 3,
    patience_early_stopping: int = 7,
    model_save_path: Path = DEFAULT_MODEL_PATH,
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Treina o modelo CNN com otimizador Adam, ReduceLROnPlateau e Early Stopping.

    Acelera o treinamento na GPU Intel Arc 140V (DirectML) ou CPU conforme o dispositivo.
    """
    if device is None:
        device, _ = get_device()

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=patience_lr,
        min_lr=1e-5,
    )

    history: dict[str, list[float]] = {
        "accuracy": [],
        "loss": [],
        "val_accuracy": [],
        "val_loss": [],
    }

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    logger.info(
        f"Iniciando treinamento no dispositivo [{device}]: epochs={epochs}, lr={learning_rate}"
    )

    for epoch in range(1, epochs + 1):
        # Modo de Treinamento
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Época {epoch:02d}/{epochs:02d} [Treino]", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Modo de Validação
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = correct_val / total_val

        # Registra histórico
        history["loss"].append(epoch_train_loss)
        history["accuracy"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_accuracy"].append(epoch_val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Época {epoch:02d}/{epochs:02d} | "
            f"Treino Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc * 100:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc * 100:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        scheduler.step(epoch_val_loss)

        # Early Stopping e salvamento do melhor modelo
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0

            model_save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(best_weights, model_save_path)
            logger.success(f"-> Melhor modelo salvo com val_loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience_early_stopping:
                logger.warning(
                    f"Early Stopping acionado após {epoch} épocas sem melhora em val_loss."
                )
                break

    # Restaura melhores pesos
    model.load_state_dict(best_weights)
    logger.success(f"Treinamento finalizado! Modelo final salvo em: {model_save_path}")
    return model, history


@app.command()
def main(
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    model_path: Path = DEFAULT_MODEL_PATH,
):
    """CLI para treinar o modelo na GPU Intel Arc 140V."""
    device, _ = get_device()
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)
    model = create_cifar10_cnn(device=device)
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        model_save_path=model_path,
    )


if __name__ == "__main__":
    app()
