import math
from collections import Counter
from pandas import DataFrame, Series
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

    ganhos_informacao_validos = []
    candidatos_de_divisao = []

    features_cols = [col for col in dados_df.columns if col != 'label']

    for feature_name in features_cols:
        valores_unicos_ordenados = sorted(set(dados_df[feature_name]))
        pontos_de_corte = []
        if len(valores_unicos_ordenados) > 1:
            for i in range(len(valores_unicos_ordenados) - 1):
                ponto_medio = (valores_unicos_ordenados[i] + valores_unicos_ordenados[i + 1]) / 2
                pontos_de_corte.append(ponto_medio)
        else:
            continue
        for valor in pontos_de_corte:
            grupo_esquerda, grupo_direita = dividir_dataset(dados_df, feature_name, valor)

            if grupo_esquerda.empty or grupo_direita.empty:
                continue

            p_esquerda = len(grupo_esquerda) / len(dados_df)
            p_direita = 1.0 - p_esquerda
            entropia_ponderada = (p_esquerda * calcular_entropia(grupo_esquerda) +
                                  p_direita * calcular_entropia(grupo_direita))

            ganho_informacao = entropia_base - entropia_ponderada

            if ganho_informacao == 0:
                continue
            ganhos_informacao_validos.append(ganho_informacao)
            split_info = -p_esquerda * math.log2(p_esquerda) - p_direita * math.log2(p_direita)
            if split_info == 0:
                gain_ratio = 0
            else:
                gain_ratio = ganho_informacao / split_info
            candidatos_de_divisao.append((gain_ratio, ganho_informacao, feature_name, valor))
    if not candidatos_de_divisao:
        return {'feature': None, 'valor': None, 'ganho': 0}
    if not ganhos_informacao_validos:
        return {'feature': None, 'valor': None, 'ganho': 0}

    media_ganho_info = sum(ganhos_informacao_validos) / len(ganhos_informacao_validos)

    candidatos_filtrados = [
        candidato for candidato in candidatos_de_divisao
        if candidato[1] >= media_ganho_info
    ]
    if not candidatos_filtrados:
        candidatos_filtrados = candidatos_de_divisao

    melhor_candidato = max(candidatos_filtrados, key=lambda item: item[0])

    melhor_ganho_ratio, melhor_ganho_info, melhor_feature, melhor_valor = melhor_candidato

    return {'feature': melhor_feature, 'valor': melhor_valor, 'ganho': melhor_ganho_info}

def construir_arvore_recursivo(
        dados_df: DataFrame,
        profundidade_max: int,
        profundidade_atual: int
):
    if dados_df.empty:
        return None

    if profundidade_atual >= profundidade_max:
        return dados_df['label'].mode()[0]

    divisao = encontrar_melhor_divisao(dados_df)

    GANHO_MINIMO = 0.01
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

def construir_arvore_c4_5(treino_df: DataFrame, profundidade_max: int=10):
    return construir_arvore_recursivo(treino_df, profundidade_max, 0)


def predizer_amostra_arvore(arvore: dict, amostra_series: Series):
    if not isinstance(arvore, dict):
        return arvore
    feature_name, valor_divisao = arvore['feature'], arvore['valor']

    if amostra_series[feature_name] < valor_divisao:
        return predizer_amostra_arvore(arvore['esquerda'], amostra_series)
    else:
        return predizer_amostra_arvore(arvore['direita'], amostra_series)


if __name__ == "__main__":
    caminho_arquivo = "./files/treino_sinais_vitais_com_label.txt"
    features_para_ignorar_global = ["index", "feature_6"]
    nome_label = "label"

    dataset_completo = carregar_dados(
        caminho_arquivo,
        features_to_ignore=FEATURES_TO_IGNORE,
        label_col_name=nome_label
    )
    features_cols = [col for col in dataset_completo.columns if col != "label"]

    treino, teste = dividir_treino_teste(dataset_completo, TEST_PROPORTION)

    print("--- Treinando Árvore de Decisão (C4.5) ---")

    arvore = construir_arvore_c4_5(treino, profundidade_max=MAX_DEPTH)

    teste_features_df = teste.drop(columns=["label"])

    predicoes_arvore = []
    imprimir_arvore(arvore)
    for index, amostra_series in teste_features_df.iterrows():
        pred = predizer_amostra_arvore(arvore, amostra_series)
        predicoes_arvore.append(pred)
    acuracia_arvore = calcular_acuracia(teste, predicoes_arvore)
    print(f"Acurácia da Árvore de Decisão: {acuracia_arvore:.2f}%\n")
