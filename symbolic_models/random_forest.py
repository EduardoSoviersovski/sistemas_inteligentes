from collections import Counter
from pandas import DataFrame, Series
from symbolic_models.decision_tree_id3 import construir_arvore_id3, predizer_amostra_arvore
from symbolic_models.utils import carregar_dados, dividir_treino_teste, calcular_acuracia


def criar_amostra_bootstrap(dataset_df: DataFrame):
    return dataset_df.sample(n=len(dataset_df), replace=True, ignore_index=True)


def random_forest_treino(treino_df: DataFrame, n_arvores: int, profundidade_max: int,
                         indices_para_ignorar: list = None):
    floresta = []
    indices_globais = set(indices_para_ignorar or [])

    for _ in range(n_arvores):
        amostra_bootstrap = criar_amostra_bootstrap(treino_df)
        arvore = construir_arvore_id3(
            amostra_bootstrap,
            profundidade_max=profundidade_max,
            indices_para_ignorar=indices_globais
        )
        floresta.append(arvore)
    return floresta


def random_forest_predicao(floresta: list, amostra_series: Series):
    predicoes = [predizer_amostra_arvore(arvore, amostra_series) for arvore in floresta]

    return Counter(predicoes).most_common(1)[0][0]


if __name__ == '__main__':
    caminho_arquivo = './files/treino_sinais_vitais_com_label.txt'
    features_para_ignorar_global = ["index"]
    nome_label = "label"

    dataset_completo = carregar_dados(
        caminho_arquivo,
        features_to_ignore=features_para_ignorar_global,
        label_col_name=nome_label
    )

    features_cols = [col for col in dataset_completo.columns if col != nome_label]
    min_vals = dataset_completo[features_cols].min()
    max_vals = dataset_completo[features_cols].max()

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1

    dataset_completo[features_cols] = (dataset_completo[features_cols] - min_vals) / range_vals

    treino, teste = dividir_treino_teste(dataset_completo, 0.3)

    print("--- Treinando Random Forest (ID3) ---")
    floresta = random_forest_treino(
        treino,
        n_arvores=10,
        profundidade_max=5,
        indices_para_ignorar=features_para_ignorar_global
    )

    teste_features_df = teste.drop(columns=[nome_label])

    predicoes_rf = []
    for index, amostra_series in teste_features_df.iterrows():
        pred = random_forest_predicao(floresta, amostra_series)
        predicoes_rf.append(pred)

    acuracia_rf = calcular_acuracia(teste, predicoes_rf)
    print(f"Acurácia do Random Forest: {acuracia_rf:.2f}%\n")
