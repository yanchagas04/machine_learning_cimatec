from pathlib import Path
from typing import Any

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import typer

from classificador_imagens.config import CLASS_NAMES, FIGURES_DIR

app = typer.Typer()


def plot_sample_images(
    x: np.ndarray,
    y: np.ndarray,
    class_names: list[str] | None = None,
    num_images: int = 9,
    save_path: str | Path | None = FIGURES_DIR / "dataset_samples.png",
    show: bool = False,
) -> Path:
    """Plota uma grade com exemplos de imagens do dataset acompanhadas de suas classes reais."""
    if class_names is None:
        class_names = CLASS_NAMES

    fig = plt.figure(figsize=(10, 10))
    for i in range(min(num_images, len(x))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x[i])
        plt.title(class_names[int(y[i])])
        plt.axis("off")

    plt.suptitle("Exemplos da base CIFAR-10", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Gráfico de exemplos salvo em: {save_path}")

    if show:
        plt.show()
    plt.close(fig)

    return Path(save_path) if save_path else None


def plot_training_history(
    history: Any,
    save_path: str | Path | None = FIGURES_DIR / "training_history.png",
    show: bool = False,
) -> Path:
    """Plota as curvas de Acurácia e Loss ao longo das épocas (Treino vs Validação)."""
    # Suporta objeto History do Keras ou dicionário history.history
    hist_dict = history.history if hasattr(history, "history") else history

    fig = plt.figure(figsize=(14, 5))

    # Gráfico de Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(hist_dict.get("accuracy", []), label="Treino", linewidth=2)
    if "val_accuracy" in hist_dict:
        plt.plot(hist_dict["val_accuracy"], label="Validação", linewidth=2)
    plt.title("Acurácia por Época", fontsize=12, fontweight="bold")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Gráfico de Loss
    plt.subplot(1, 2, 2)
    plt.plot(hist_dict.get("loss", []), label="Treino", linewidth=2)
    if "val_loss" in hist_dict:
        plt.plot(hist_dict["val_loss"], label="Validação", linewidth=2)
    plt.title("Loss por Época", fontsize=12, fontweight="bold")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Gráfico do histórico de treino salvo em: {save_path}")

    if show:
        plt.show()
    plt.close(fig)

    return Path(save_path) if save_path else None


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] | None = None,
    save_path: str | Path | None = FIGURES_DIR / "confusion_matrix.png",
    show: bool = False,
) -> Path:
    """Plota as matrizes de confusão (Contagem Absoluta e Normalizada lado a lado)."""
    if class_names is None:
        class_names = CLASS_NAMES

    fig, ax = plt.subplots(1, 2, figsize=(18, 7))

    # 1. Contagens Absolutas
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax[0],
    )
    ax[0].set_title("Matriz de Confusão (Contagem Absoluta)", fontsize=13, fontweight="bold")
    ax[0].set_xlabel("Classe Predita", fontsize=11)
    ax[0].set_ylabel("Classe Real", fontsize=11)
    ax[0].tick_params(axis="x", rotation=45)

    # 2. Normalizada (% de acerto por classe real)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax[1],
    )
    ax[1].set_title("Matriz de Confusão Normalizada (%)", fontsize=13, fontweight="bold")
    ax[1].set_xlabel("Classe Predita", fontsize=11)
    ax[1].set_ylabel("Classe Real", fontsize=11)
    ax[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Matriz de confusão salva em: {save_path}")

    if show:
        plt.show()
    plt.close(fig)

    return Path(save_path) if save_path else None


def plot_prediction_samples(
    x_test: np.ndarray,
    y_test: np.ndarray,
    pred_labels: np.ndarray,
    class_names: list[str] | None = None,
    num_images: int = 9,
    save_path: str | Path | None = FIGURES_DIR / "prediction_samples.png",
    show: bool = False,
) -> Path:
    """Visualização de amostras do teste com rótulo real vs predito (Verde=Acerto | Vermelho=Erro)."""
    if class_names is None:
        class_names = CLASS_NAMES

    fig = plt.figure(figsize=(10, 10))
    for i in range(min(num_images, len(x_test))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x_test[i])
        real = class_names[int(y_test[i])]
        pred = class_names[int(pred_labels[i])]

        cor = "green" if real == pred else "red"
        plt.title(f"Real: {real}\nPredição: {pred}", color=cor, fontsize=11, fontweight="bold")
        plt.axis("off")

    plt.suptitle(
        "Exemplos de Predições (Verde = Correto | Vermelho = Incorreto)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Gráfico de predições salvo em: {save_path}")

    if show:
        plt.show()
    plt.close(fig)

    return Path(save_path) if save_path else None


@app.command()
def main():
    logger.info("Módulo de plotagem carregado com sucesso.")


if __name__ == "__main__":
    app()
