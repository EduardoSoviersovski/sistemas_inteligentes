import math
from collections import Counter

from pandas import DataFrame, Series

from symbolic_models.utils import carregar_dados, dividir_treino_teste

TEST_PROPORTION = 0.3
MAX_DEPTH = 10
FEATURES_TO_IGNORE = ["index", "feature_6"]

def calcular_entropia(dados_df: DataFrame) -> float:
    if dados_df.empty:
        return 0

    contagem_labels = Counter(dados_df['label'])
    entropia = 0.0
    total_amostras = len(dados_df)

    for label in contagem_labels:
        probabilidade = contagem_labels[label] / total_amostras
        entropia -= probabilidade * math.log2(probabilidade)
    return entropia

def dividir_dataset(dados_df: DataFrame, feature_name: str, valor: float) -> tuple[Series, Series]:
    grupo_esquerda = dados_df[dados_df[feature_name] < valor]
    grupo_direita = dados_df[dados_df[feature_name] >= valor]
    return grupo_esquerda, grupo_direita

def predizer_amostra_arvore(arvore: dict, amostra_series: Series):
    if not isinstance(arvore, dict):
        return arvore
    feature_name, valor_divisao = arvore['feature'], arvore['valor']

    if amostra_series[feature_name] < valor_divisao:
        return predizer_amostra_arvore(arvore['esquerda'], amostra_series)
    else:
        return predizer_amostra_arvore(arvore['direita'], amostra_series)

def get_test_and_train_dataframes(path: str, label: str):
    dataset_completo = carregar_dados(
        path,
        features_to_ignore=FEATURES_TO_IGNORE,
        label_col_name=label
    )
    return dividir_treino_teste(dataset_completo, TEST_PROPORTION)