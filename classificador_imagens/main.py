from pathlib import Path
import sys
from typing import Annotated

# Garante que o diretório raiz do projeto esteja no sys.path
PROJ_ROOT = Path(__file__).resolve().parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from loguru import logger
import typer

from classificador_imagens.config import (
    CLASS_NAMES,
    DEFAULT_MODEL_PATH,
    FIGURES_DIR,
    get_device,
)
from classificador_imagens.dataset import get_dataloaders, load_cifar10_numpy
from classificador_imagens.modeling import (
    create_cifar10_cnn,
    evaluate_model,
    load_trained_model,
    train_model,
)
from classificador_imagens.plots import (
    plot_confusion_matrix,
    plot_prediction_samples,
    plot_sample_images,
    plot_training_history,
)

app = typer.Typer(
    help="Pipeline completo do Classificador de Imagens CIFAR-10 acelerado na GPU Intel Arc 140V."
)


def run_pipeline(
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    train: bool = True,
    evaluate: bool = True,
    plot_samples: bool = True,
    save_plots: bool = True,
    show_plots: bool = False,
    model_path: Path = DEFAULT_MODEL_PATH,
):
    """Executa o pipeline modular com aceleração automática via DirectML / Intel Arc 140V."""
    device, device_name = get_device()

    logger.info("==================================================")
    logger.info("  INICIANDO PIPELINE - CLASSIFICADOR CIFAR-10     ")
    logger.info(f"  Dispositivo: {device_name}")
    logger.info("==================================================")

    # 1. Carregamento dos dados
    logger.info("--- ETAPA 1: Carregando e Normalizando Dataset ---")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=batch_size)
    (x_train, y_train), (x_test, _) = load_cifar10_numpy()

    # 2. Visualização de amostras do dataset
    if plot_samples:
        logger.info("--- ETAPA 2: Gerando visualização de amostras do dataset ---")
        samples_path = FIGURES_DIR / "dataset_samples.png" if save_plots else None
        plot_sample_images(
            x=x_train,
            y=y_train,
            class_names=CLASS_NAMES,
            num_images=9,
            save_path=samples_path,
            show=show_plots,
        )

    # 3. Treinamento ou carregamento do modelo
    model = None
    if train:
        logger.info(f"--- ETAPA 3: Treinando Modelo CNN na GPU [{device_name}] ---")
        model = create_cifar10_cnn(device=device)

        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            model_save_path=model_path,
        )

        # Plot das curvas de acurácia e perda
        logger.info("--- ETAPA 4: Gerando gráficos do histórico de treinamento ---")
        history_path = FIGURES_DIR / "training_history.png" if save_plots else None
        plot_training_history(
            history=history,
            save_path=history_path,
            show=show_plots,
        )
    else:
        logger.info("--- ETAPA 3: Carregando Modelo Pré-treinado Existente ---")
        model = load_trained_model(model_path=model_path, device=device)

    # 4. Avaliação e Métricas de Teste
    if evaluate and model is not None:
        logger.info("--- ETAPA 5: Avaliação no Conjunto de Teste ---")
        eval_results = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            class_names=CLASS_NAMES,
        )

        logger.info("--- ETAPA 6: Gerando Matriz de Confusão e Relatórios ---")
        cm_path = FIGURES_DIR / "confusion_matrix.png" if save_plots else None
        plot_confusion_matrix(
            cm=eval_results["confusion_matrix"],
            class_names=CLASS_NAMES,
            save_path=cm_path,
            show=show_plots,
        )

        logger.info("--- ETAPA 7: Gerando Gráfico de Predições em Amostras de Teste ---")
        preds_path = FIGURES_DIR / "prediction_samples.png" if save_plots else None
        plot_prediction_samples(
            x_test=x_test[:9],
            y_test=eval_results["y_true"][:9],
            pred_labels=eval_results["y_pred"][:9],
            class_names=CLASS_NAMES,
            num_images=9,
            save_path=preds_path,
            show=show_plots,
        )

    logger.success("==================================================")
    logger.success("  PIPELINE EXECUTADO COM SUCESSO!                 ")
    logger.success(f"  Dispositivo Utilizado: {device_name}")
    logger.success(f"  Modelo salvo em:       {model_path}")
    logger.success(f"  Figuras salvas em:     {FIGURES_DIR}")
    logger.success("==================================================")


@app.command()
def main(
    epochs: Annotated[
        int, typer.Option("--epochs", "-e", help="Número de épocas de treinamento.")
    ] = 30,
    batch_size: Annotated[
        int, typer.Option("--batch-size", "-b", help="Tamanho do lote de treino e avaliação.")
    ] = 64,
    learning_rate: Annotated[
        float, typer.Option("--lr", help="Taxa de aprendizado inicial.")
    ] = 0.001,
    train: Annotated[
        bool,
        typer.Option(
            "--train/--no-train", help="Se deve treinar o modelo ou carregar um pré-treinado."
        ),
    ] = True,
    evaluate: Annotated[
        bool,
        typer.Option("--evaluate/--no-evaluate", help="Se deve avaliar no conjunto de teste."),
    ] = True,
    plot_samples: Annotated[
        bool,
        typer.Option(
            "--plot-samples/--no-plot-samples",
            help="Se deve gerar o grid com exemplos do dataset.",
        ),
    ] = True,
    save_plots: Annotated[
        bool,
        typer.Option(
            "--save-plots/--no-save-plots",
            help="Se deve salvar figuras geradas em reports/figures.",
        ),
    ] = True,
    show_plots: Annotated[
        bool,
        typer.Option(
            "--show-plots/--no-show-plots",
            help="Se deve exibir janelas interativas do matplotlib.",
        ),
    ] = False,
    model_path: Annotated[
        Path,
        typer.Option("--model-path", "-m", help="Caminho do arquivo do modelo (.pt)."),
    ] = DEFAULT_MODEL_PATH,
):
    """Ponto de entrada principal para executar o pipeline completo."""
    run_pipeline(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train=train,
        evaluate=evaluate,
        plot_samples=plot_samples,
        save_plots=save_plots,
        show_plots=show_plots,
        model_path=model_path,
    )


if __name__ == "__main__":
    app()
