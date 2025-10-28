from collections import Counter
from pandas import DataFrame, Series

from machine_learning.commons import get_test_and_train_dataframes, class_labels
from machine_learning.decision_tree_id3 import predizer_amostra_arvore, construir_arvore_recursivo
from machine_learning.utils import calcular_acuracia, carregar_dados

NUM_TREES = 10
TEST_PROPORTION = 0.3
MAX_DEPTH = 5
LABEL = "classe"
FEATURES_TO_IGNORE = ["index", "gravidade", "pSist", "pDiast"]

def criar_amostra_bootstrap(dataset_df: DataFrame):
    return dataset_df.sample(n=len(dataset_df), replace=True, ignore_index=True)

def random_forest_treino(treino_df: DataFrame, n_arvores: int, profundidade_max: int):
    floresta = []

    for _ in range(n_arvores):
        amostra_bootstrap = criar_amostra_bootstrap(treino_df)
        arvore = construir_arvore_recursivo(
            amostra_bootstrap,
            profundidade_max=profundidade_max,
            profundidade_atual=0
        )
        floresta.append(arvore)
    return floresta


def random_forest_predicao(floresta: list, amostra_series: Series):
    predicoes = [predizer_amostra_arvore(arvore, amostra_series) for arvore in floresta]

    return Counter(predicoes).most_common(1)[0][0]


if __name__ == '__main__':
    dataset_completo = carregar_dados(
        "./files/treino_sinais_vitais_com_label.txt",
        features_to_ignore=FEATURES_TO_IGNORE,
        label_col_name=LABEL
    )
    labels_unicos = dataset_completo[LABEL].unique()
    labels_ordenados = sorted(labels_unicos)
    treino, teste = get_test_and_train_dataframes(
        "./files/treino_sinais_vitais_com_label.txt",
        LABEL,
        features_to_ignore=FEATURES_TO_IGNORE,
        test_proportion=TEST_PROPORTION
    )

    print("--- Treinando Random Forest (ID3) ---")
    floresta = random_forest_treino(
        treino,
        n_arvores=NUM_TREES,
        profundidade_max=MAX_DEPTH
    )

    teste_features_df = teste.drop(columns=[LABEL])

    predicoes_rf = []
    for index, amostra_series in teste_features_df.iterrows():
        pred = random_forest_predicao(floresta, amostra_series)
        predicoes_rf.append(pred)

    class_labels(predicoes_rf, teste, LABEL, labels_ordenados)

    acuracia_rf = calcular_acuracia(teste, predicoes_rf)
    print(f"Acurácia do Random Forest: {acuracia_rf:.2f}%\n")
