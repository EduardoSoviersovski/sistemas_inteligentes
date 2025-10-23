import shutil

from pandas import DataFrame

from symbolic_models.decision_tree_commons import dividir_dataset, calcular_entropia, \
    get_test_and_train_dataframes, predizer_amostra_arvore, visualizar_arvore_customizada
from symbolic_models.utils import calcular_acuracia, imprimir_arvore

TEST_PROPORTION = 0.3
MAX_DEPTH = 5
FEATURES_TO_IGNORE = ["index"]

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

if __name__ == "__main__":
    treino, teste = get_test_and_train_dataframes(
        "./files/treino_sinais_vitais_com_label.txt",
        "label",
        features_to_ignore=FEATURES_TO_IGNORE,
        test_proportion=TEST_PROPORTION
    )

    print("--- Treinando Árvore de Decisão (ID3) ---")
    arvore = construir_arvore_recursivo(treino, MAX_DEPTH, 0)

    teste_features_df = teste.drop(columns=["label"])

    predicoes_arvore = []

    for index, amostra_series in teste_features_df.iterrows():
        pred = predizer_amostra_arvore(arvore, amostra_series)
        predicoes_arvore.append(pred)
    acuracia_arvore = calcular_acuracia(teste, predicoes_arvore)
    if shutil.which("dot") is not None:
        visualizar_arvore_customizada(arvore, nome_arquivo='minha_arvore_id3')
    else:
        imprimir_arvore(arvore)
    print(f"Acurácia da Árvore de Decisão: {acuracia_arvore:.2f}%\n")
