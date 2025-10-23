import math
from collections import Counter
from pandas import DataFrame
from symbolic_models.utils import carregar_dados, dividir_treino_teste, calcular_acuracia, imprimir_arvore

TEST_PROPORTION = 0.3
MAX_DEPTH = 5
FEATURES_TO_IGNORE = ["index"]

def calcular_entropia(dados_df: DataFrame):
    if dados_df.empty:
        return 0

    contagem_labels = Counter(dados_df['label'])
    entropia = 0.0
    total_amostras = len(dados_df)

    for label in contagem_labels:
        probabilidade = contagem_labels[label] / total_amostras
        entropia -= probabilidade * math.log2(probabilidade)
    return entropia

def dividir_dataset(dados_df: DataFrame, feature_name: str, valor: float):
    grupo_esquerda = dados_df[dados_df[feature_name] < valor]
    grupo_direita = dados_df[dados_df[feature_name] >= valor]
    return grupo_esquerda, grupo_direita

def encontrar_melhor_divisao(dados_df: DataFrame):
    entropia_base = calcular_entropia(dados_df)
    melhor_ganho = 0.0
    melhor_feature = None
    melhor_valor = None

    features_cols = [col for col in dados_df.columns if col != 'label']

    for feature_name in features_cols:
        valores_unicos = set(dados_df[feature_name])
        for valor in valores_unicos:
            grupo_esquerda, grupo_direita = dividir_dataset(dados_df, feature_name, valor)

            if grupo_esquerda.empty or grupo_direita.empty:
                continue

            p_esquerda = len(grupo_esquerda) / len(dados_df)
            entropia_ponderada = (p_esquerda * calcular_entropia(grupo_esquerda) + (1 - p_esquerda) * calcular_entropia(grupo_direita))

            ganho_informacao = entropia_base - entropia_ponderada

            if ganho_informacao > melhor_ganho:
                melhor_ganho = ganho_informacao
                melhor_feature = feature_name
                melhor_valor = valor

    return {'feature': melhor_feature, 'valor': melhor_valor, 'ganho': melhor_ganho}

def construir_arvore_recursivo(dados_df: DataFrame, profundidade_max: int, profundidade_atual: int):
    if dados_df.empty:
        return None

    if profundidade_atual >= profundidade_max:
        return dados_df['label'].mode()[0]

    divisao = encontrar_melhor_divisao(dados_df)

    GANHO_MINIMO = 0.1
    if divisao['ganho'] < GANHO_MINIMO or divisao['feature'] is None:
        return dados_df['label'].mode()[0]

    feature_escolhida = divisao['feature']

    grupo_esquerda, grupo_direita = dividir_dataset(dados_df, feature_escolhida, divisao['valor'])

    arvore = {
        'feature': feature_escolhida,
        'valor': divisao['valor'],
        'esquerda': construir_arvore_recursivo(
            grupo_esquerda,
            profundidade_max,
            profundidade_atual + 1
        ),
        'direita': construir_arvore_recursivo(
            grupo_direita,
            profundidade_max,
            profundidade_atual + 1
        )
    }

    return arvore

def construir_arvore_id3(treino_df: DataFrame, profundidade_max=10):
    return construir_arvore_recursivo(treino_df, profundidade_max, 0)


def predizer_amostra_arvore(arvore, amostra_series):
    if not isinstance(arvore, dict):
        return arvore
    feature_name, valor_divisao = arvore['feature'], arvore['valor']

    if amostra_series[feature_name] < valor_divisao:
        return predizer_amostra_arvore(arvore['esquerda'], amostra_series)
    else:
        return predizer_amostra_arvore(arvore['direita'], amostra_series)


if __name__ == "__main__":
    caminho_arquivo = "./files/treino_sinais_vitais_com_label.txt"

    nome_label = "label"

    dataset_completo = carregar_dados(
        caminho_arquivo,
        features_to_ignore=FEATURES_TO_IGNORE,
        label_col_name=nome_label
    )
    features_cols = [col for col in dataset_completo.columns if col != "label"]

    treino, teste = dividir_treino_teste(dataset_completo, TEST_PROPORTION)

    print("--- Treinando Árvore de Decisão (ID3) ---")

    arvore = construir_arvore_id3(treino, profundidade_max=MAX_DEPTH)

    teste_features_df = teste.drop(columns=["label"])

    predicoes_arvore = []
    imprimir_arvore(arvore)
    for index, amostra_series in teste_features_df.iterrows():
        pred = predizer_amostra_arvore(arvore, amostra_series)
        predicoes_arvore.append(pred)
    acuracia_arvore = calcular_acuracia(teste, predicoes_arvore)
    print(f"Acurácia da Árvore de Decisão: {acuracia_arvore:.2f}%\n")
