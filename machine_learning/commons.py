import math
import random
from collections import Counter

import graphviz
from pandas import DataFrame, Series
from sklearn.metrics import classification_report, confusion_matrix

from machine_learning.utils import carregar_dados, dividir_treino_teste

def calcular_entropia(dados_df: DataFrame) -> float:
    if dados_df.empty:
        return 0

    contagem_labels = Counter(dados_df['classe'])
    entropia = 0.0
    total_amostras = len(dados_df)

    for label in contagem_labels:
        probabilidade = contagem_labels[label] / total_amostras
        entropia -= probabilidade * math.log2(probabilidade)
    return entropia

def dividir_dataset(dados_df: DataFrame, feature_name: str, valor: float) -> tuple[DataFrame, DataFrame]:
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

def get_test_and_train_dataframes(path: str, label: str, features_to_ignore: list[str], test_proportion: float) -> tuple[DataFrame, DataFrame]:
    dataset_completo = carregar_dados(
        path,
        features_to_ignore=features_to_ignore,
        label_col_name=label
    )
    return dividir_treino_teste(dataset_completo, test_proportion)

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

def visualizar_rede_neural(nn, feature_names, class_labels, nome_arquivo='rede_neural'):
    dot = graphviz.Digraph(comment='Rede Neural')
    dot.attr(rankdir='LR', nodesep='0.3', ranksep='5.0')


    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Camada de Entrada', rank='same', style='filled', color='lightgray')
        c.attr(rank='min')
        for i in range(nn.n_entradas):
            label = feature_names[i] if i < len(feature_names) else f'Entrada {i + 1}'
            c.node(f'in_{i}', label=label, shape='box', style='filled', color='white')

    with dot.subgraph(name='cluster_hidden') as c:
        c.attr(label='Camada Oculta', rank='same', style='filled', color='lightgray')
        for j in range(nn.n_ocultas):
            bias = nn.bias_oculta[j]
            c.node(f'h_{j}', label=f'H{j}\n(Bias: {bias:.2f})', shape='circle', style='filled', color='white')

    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Camada de Saída', rank='same', style='filled', color='lightgray')
        for k in range(nn.n_saidas):
            label = class_labels[k] if k < len(class_labels) else f'Saída {k + 1}'
            bias = nn.bias_saida[k]
            c.node(f'out_{k}', label=f'Classe: {label}\n(Bias: {bias:.2f})', shape='doublecircle', style='filled',
                   color='lightblue')

    for i in range(nn.n_entradas):
        for j in range(nn.n_ocultas):
            peso = nn.pesos_oculta[i][j]
            dot.edge(f'in_{i}', f'h_{j}', label=f'{peso:.2f}')

    for j in range(nn.n_ocultas):
        for k in range(nn.n_saidas):
            peso = nn.pesos_saida[j][k]
            dot.edge(f'h_{j}', f'out_{k}', label=f'{peso:.2f}')

    try:
        dot.render(nome_arquivo, view=True, format='png', cleanup=True)
        print(f"Visualização da rede salva como '{nome_arquivo}.png' e aberta.")
    except Exception as e:
        print(f"Erro ao renderizar com Graphviz: {e}")
        print("Verifique se o software Graphviz está instalado e no PATH do sistema.")

def random_oversample(dados_lista, idx_label=-1):
    print("Executando Random Oversample...")
    contagem_classes = {}
    amostras_por_classe = {}

    for amostra in dados_lista:
        label = amostra[idx_label]
        if label not in contagem_classes:
            contagem_classes[label] = 0
            amostras_por_classe[label] = []
        contagem_classes[label] += 1
        amostras_por_classe[label].append(amostra)

    if not contagem_classes:
        return []

    max_contagem = 0
    for label in contagem_classes:
        if contagem_classes[label] > max_contagem:
            max_contagem = contagem_classes[label]

    print(f"Contagem original: {contagem_classes}")
    print(f"Classe majoritária tem {max_contagem} amostras.")

    # 3. Criar o novo dataset balanceado
    dados_balanceados = []
    for label in contagem_classes:
        amostras_classe = amostras_por_classe[label]

        # Adiciona todas as amostras originais
        dados_balanceados.extend(amostras_classe)

        # Calcula quantas amostras sintéticas (duplicatas) precisa
        n_para_adicionar = max_contagem - contagem_classes[label]

        if n_para_adicionar > 0:
            # Adiciona duplicatas aleatórias
            novas_amostras = random.choices(amostras_classe, k=n_para_adicionar)
            dados_balanceados.extend(novas_amostras)

    print(f"Dataset balanceado com {len(dados_balanceados)} amostras.")
    random.shuffle(dados_balanceados)
    return dados_balanceados

def class_labels(predicoes_nn, teste_df, nome_label, labels_ordenados):
    y_verdadeiro = teste_df[nome_label].values.tolist()
    nomes_classes_str = [str(int(l)) for l in labels_ordenados]

    print("\n--- Relatório de Classificação (Precision, Recall, F1-Score) ---")
    report = classification_report(
        y_verdadeiro,
        predicoes_nn,
        target_names=nomes_classes_str,
        labels=labels_ordenados,
        zero_division=0
    )
    print(report)

    # --- Métrica 2: Matriz de Confusão ---
    print("\n--- Matriz de Confusão ---")
    print("(Linhas = Real, Colunas = Predito)\n")
    cm = confusion_matrix(y_verdadeiro, predicoes_nn, labels=labels_ordenados)

    # Usar Pandas para formatar a matriz de confusão de forma legível
    cm_df = DataFrame(cm,index=[f"Real: {name}" for name in nomes_classes_str],
                    columns=[f"Pred: {name}" for name in nomes_classes_str])
    print(cm_df)
