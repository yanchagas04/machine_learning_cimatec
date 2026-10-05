# Classificador de Imagens CIFAR-10 (Aceleração Intel Arc 140V)

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Projeto modular de Deep Learning para classificação de imagens do dataset **CIFAR-10**, acelerado via hardware na GPU **Intel(R) Arc(TM) 140V (16GB)** através do **DirectML** no Windows nativo.

A arquitetura implementada é uma CNN estilo VGG com:
1. **Data Augmentation Dinâmico**: Flip horizontal, rotações e cortes leves.
2. **Blocos Convolucionais Duplos**: Camadas consecutivas 32 -> 64 -> 128 filtros com `padding='same'`.
3. **Batch Normalization**: Estabilização dos gradientes e convergência acelerada.
4. **Dropout Progressivo**: 0.2 -> 0.3 -> 0.4 -> 0.5 para controle rigoroso de overfitting.
5. **Aceleração por Hardware**: Detecção automática de GPU (`Intel(R) Arc(TM) 140V GPU (16GB) (DirectML)`).
6. **Otimizador & Callbacks**: Adam com `ReduceLROnPlateau` e `EarlyStopping` (restauração dos melhores pesos).

---

## 📁 Organização do Projeto

```text
classificador_imagens/
├── main.py                     <- PONTO DE ENTRADA PRINCIPAL (orquestra todo o pipeline)
├── pyproject.toml              <- Dependências gerenciadas com uv (torch-directml, torchvision, etc.)
├── README.md                   <- Documentação principal do projeto
├── data/
│   ├── raw/                    <- Dataset CIFAR-10 em cache local (cifar-10-batches-py)
│   └── processed/              <- Dados processados
├── models/
│   └── cifar10_cnn_directml.pt <- Pesos do melhor modelo treinado na Intel Arc
├── notebooks/
│   └── classificador_imagens.ipynb <- Notebook original de exploração e validação
├── reports/
│   └── figures/                <- Gráficos gerados automaticamente
│       ├── dataset_samples.png    <- Grade 3x3 de amostras do dataset
│       ├── training_history.png   <- Curvas de Acurácia e Perda (Treino vs Validação)
│       ├── confusion_matrix.png   <- Matrizes de Confusão (Absoluta e Normalizada %)
│       └── prediction_samples.png <- Amostras preditas (Verde = Correto | Vermelho = Erro)
└── classificador_imagens/
    ├── config.py               <- Caminhos, classes do CIFAR-10 e detecção de GPU
    ├── dataset.py              <- DataLoaders PyTorch e utilitários de cache
    ├── features.py             <- Pré-processamento e redimensionamento de imagens avulsas
    ├── plots.py                <- Funções de visualização com Matplotlib e Seaborn
    └── modeling/
        ├── models.py           <- Arquitetura Cifar10CNN estilo VGG
        ├── train.py            <- Treinamento com otimização, ReduceLROnPlateau e Early Stopping
        └── predict.py          <- Avaliação de teste, classification report e inferência
```

---

## 🚀 Como Executar

No terminal, acesse a pasta do projeto:

```bash
cd classificador_imagens
```

### 1. Executar o Pipeline Completo na Intel Arc 140V

Basta executar o [main.py](file:///c:/Users/yanchagas04/Documents/dev/machine_learning_cimatec/classificador_imagens/main.py):

```bash
uv run python main.py
```

### 2. Opções da Linha de Comando (CLI)

```bash
# Treinar com 20 épocas e lote de 128
uv run python main.py --epochs 20 --batch-size 128

# Apenas avaliar um modelo já treinado (sem re-treinar)
uv run python main.py --no-train --evaluate

# Teste rápido (1 época)
uv run python main.py --epochs 1 --batch-size 128

# Consultar todas as opções disponíveis
uv run python main.py --help
```
