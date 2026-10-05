from pathlib import Path

from loguru import logger
import numpy as np
from tensorflow.keras.utils import img_to_array, load_img
import typer

app = typer.Typer()


def preprocess_image(
    image: np.ndarray | str | Path,
    target_size: tuple[int, int] = (32, 32),
) -> np.ndarray:
    """Prepara uma única imagem para inferência no modelo CNN.

    Args:
        image: Caminho para o arquivo de imagem (str ou Path) ou array numpy.
        target_size: Dimensões (altura, largura) esperadas pelo modelo (default: (32, 32)).

    Returns:
        Array numpy com formato (1, 32, 32, 3) e valores normalizados em [0, 1].
    """
    if isinstance(image, (str, Path)):
        img = load_img(str(image), target_size=target_size)
        arr = img_to_array(img)
    elif isinstance(image, np.ndarray):
        arr = image.copy()
        if arr.ndim == 2:  # Grayscale to RGB
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[-1] == 4:  # RGBA to RGB
            arr = arr[..., :3]
    else:
        raise TypeError(f"Tipo de imagem não suportado: {type(image)}")

    if arr.max() > 1.0:
        arr = arr.astype("float32") / 255.0
    else:
        arr = arr.astype("float32")

    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)

    return arr


def preprocess_batch(images: np.ndarray) -> np.ndarray:
    """Garante que um lote de imagens esteja no formato float32 [0, 1]."""
    arr = images.copy()
    if arr.max() > 1.0:
        arr = arr.astype("float32") / 255.0
    return arr.astype("float32")


@app.command()
def main():
    logger.info("Módulo de features/pré-processamento de imagens carregado com sucesso.")


if __name__ == "__main__":
    app()
