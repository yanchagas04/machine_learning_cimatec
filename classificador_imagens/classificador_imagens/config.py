from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Default Model and Artifact Paths
DEFAULT_MODEL_PATH = MODELS_DIR / "cifar10_cnn_directml.pt"

# Create directories if they do not exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# CIFAR-10 Class Names
CLASS_NAMES = [
    "avião",
    "automóvel",
    "pássaro",
    "gato",
    "cervo",
    "cachorro",
    "sapo",
    "cavalo",
    "navio",
    "caminhão",
]


def get_device():
    """Detecta automaticamente o dispositivo de aceleração.

    Prioriza DirectML (Intel Arc 140V), XPU, CUDA e fallback para CPU.
    """
    try:
        import torch_directml

        device = torch_directml.device()
        name = torch_directml.device_name(0)
        logger.info(f"Aceleração por GPU ativada: {name} (DirectML)")
        return device, name
    except (ImportError, RuntimeError, AttributeError) as exc:
        logger.debug(f"DirectML não disponível: {exc}")

    try:
        import torch

        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")
            name = torch.xpu.get_device_name(0)
            logger.info(f"Aceleração por GPU ativada: {name} (Intel XPU)")
            return device, name
        if torch.cuda.is_available():
            device = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            logger.info(f"Aceleração por GPU ativada: {name} (CUDA)")
            return device, name
    except (ImportError, RuntimeError, AttributeError) as exc:
        logger.debug(f"XPU/CUDA não disponível: {exc}")

    import torch

    logger.warning("Nenhuma GPU detectada. Utilizando CPU para processamento.")
    return torch.device("cpu"), "CPU"


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
