from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch import nn
from torch.utils.data import DataLoader
import typer

from classificador_imagens.config import CLASS_NAMES, DEFAULT_MODEL_PATH, get_device
from classificador_imagens.dataset import get_dataloaders
from classificador_imagens.features import preprocess_image
from classificador_imagens.modeling.models import Cifar10CNN, create_cifar10_cnn

app = typer.Typer()


def load_trained_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    device: torch.device | None = None,
) -> Cifar10CNN:
    """Carrega os pesos salvos do modelo Cifar10CNN."""
    if device is None:
        device, _ = get_device()

    if not model_path.exists():
        raise FileNotFoundError(f"Arquivo de modelo não encontrado: {model_path}")

    logger.info(f"Carregando modelo de: {model_path} no dispositivo [{device}]")
    model = create_cifar10_cnn(device=device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    logger.success("Modelo carregado com sucesso.")
    return model


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Avalia o modelo no conjunto de teste na GPU Intel Arc 140V."""
    if device is None:
        device, _ = get_device()
    if class_names is None:
        class_names = CLASS_NAMES

    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    logger.info("Avaliando modelo no conjunto de teste...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    test_loss = running_loss / len(y_true)
    test_acc = float((y_pred == y_true).mean())

    logger.info(f"Acurácia no teste: {(test_acc * 100):.2f}%")
    logger.info(f"Loss no teste: {test_loss:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    logger.info("\n--- Relatório de Classificação por Classe ---\n" + report)

    return {
        "loss": float(test_loss),
        "accuracy": float(test_acc),
        "y_pred": y_pred,
        "y_true": y_true,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def predict_single_image(
    model: nn.Module,
    image: str | Path | np.ndarray,
    device: torch.device | None = None,
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Executa inferência em uma única imagem."""
    if device is None:
        device, _ = get_device()
    if class_names is None:
        class_names = CLASS_NAMES

    # preprocess_image returns (1, 32, 32, 3) numpy array
    arr = preprocess_image(image)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])
    class_name = class_names[class_id]

    logger.info(f"Predição: {class_name} ({confidence * 100:.2f}%)")
    return {
        "class_id": class_id,
        "class_name": class_name,
        "confidence": confidence,
        "probabilities": probs,
    }


@app.command()
def main(
    model_path: Path = DEFAULT_MODEL_PATH,
    batch_size: int = 64,
):
    """CLI para avaliar o modelo no teste."""
    device, _ = get_device()
    model = load_trained_model(model_path=model_path, device=device)
    _, _, test_loader = get_dataloaders(batch_size=batch_size)
    evaluate_model(model=model, test_loader=test_loader, device=device)


if __name__ == "__main__":
    app()
