import math
from collections import Counter

import graphviz
from pandas import DataFrame, Series

from symbolic_models.utils import carregar_dados, dividir_treino_teste

TEST_PROPORTION = 0.3
MAX_DEPTH = 5
FEATURES_TO_IGNORE = ["index"]

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

def _gerar_dot_recursivo(arvore, dot, node_id_counter):
    current_id = str(node_id_counter[0])
    node_id_counter[0] += 1

    if not isinstance(arvore, dict):
        dot.node(
            current_id,
            label=f"Classe: {arvore}",
            shape='box',
            style='filled',
            color='lightblue'
        )
        return current_id

    feature = arvore['feature']
    valor = arvore['valor']

    label = f"{feature}\n < {valor:.2f}?"

    dot.node(
        current_id,
        label=label,
        shape='ellipse',
        style='filled',
        color='lightgray'
    )

    id_esquerda = _gerar_dot_recursivo(arvore['esquerda'], dot, node_id_counter)
    dot.edge(current_id, id_esquerda, label='Verdadeiro')

    id_direita = _gerar_dot_recursivo(arvore['direita'], dot, node_id_counter)
    dot.edge(current_id, id_direita, label='Falso')

    return current_id


def visualizar_arvore_customizada(arvore, nome_arquivo='arvore_customizada'):
    dot = graphviz.Digraph(comment='Árvore de Decisão Customizada')
    dot.attr(rankdir='TB')
    node_id_counter = [0]

    _gerar_dot_recursivo(arvore, dot, node_id_counter)

    try:
        dot.render(nome_arquivo, view=True, format='png', cleanup=True)
        print(f"Visualização da árvore salva como '{nome_arquivo}.png' e aberta.")
    except Exception as e:
        print(f"Erro ao renderizar com Graphviz: {e}")
        print("Verifique se o software Graphviz está instalado e no PATH do sistema.")