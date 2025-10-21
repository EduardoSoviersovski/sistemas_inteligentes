import math
from collections import Counter

from utils import carregar_dados, dividir_treino_teste, calcular_acuracia


def calcular_entropia(dados):
    if not dados:
        return 0
    contagem_labels = Counter(linha[-1] for linha in dados)
    entropia = 0.0
    total_amostras = len(dados)
    for label in contagem_labels:
        probabilidade = contagem_labels[label] / total_amostras
        entropia -= probabilidade * math.log2(probabilidade)
    return entropia

def dividir_dataset(dados, indice_feature, valor):
    grupo_esquerda = [linha for linha in dados if linha[indice_feature] < valor]
    grupo_direita = [linha for linha in dados if linha[indice_feature] >= valor]
    return grupo_esquerda, grupo_direita

def encontrar_melhor_divisao(dados):
    entropia_base = calcular_entropia(dados)
    melhor_ganho = 0.0
    melhor_feature = None
    melhor_valor = None
    
    num_features = len(dados[0]) - 1
    
    for i in range(num_features):
        valores_unicos = set(linha[i] for linha in dados)
        for valor in valores_unicos:
            grupo_esquerda, grupo_direita = dividir_dataset(dados, i, valor)
            
            if not grupo_esquerda or not grupo_direita:
                continue
            
            p_esquerda = len(grupo_esquerda) / len(dados)
            entropia_ponderada = (p_esquerda * calcular_entropia(grupo_esquerda) +
                                  (1 - p_esquerda) * calcular_entropia(grupo_direita))
            
            ganho_informacao = entropia_base - entropia_ponderada
            
            if ganho_informacao > melhor_ganho:
                melhor_ganho = ganho_informacao
                melhor_feature = i
                melhor_valor = valor
                
    return {'feature': melhor_feature, 'valor': melhor_valor, 'ganho': melhor_ganho}

def no_terminal(grupo):
    labels = [linha[-1] for linha in grupo]
    return Counter(labels).most_common(1)[0][0]

def construir_arvore_recursivo(dados, profundidade_max, profundidade_atual):
    """Função recursiva para construir a árvore de decisão."""
    if not dados:
        return None

    # Critérios de parada
    labels = [linha[-1] for linha in dados]
    if len(set(labels)) == 1:
        return labels[0]
        
    if profundidade_atual >= profundidade_max:
        return no_terminal(dados)

    divisao = encontrar_melhor_divisao(dados)

    if divisao['ganho'] == 0:
        return no_terminal(dados)

    grupo_esquerda, grupo_direita = dividir_dataset(dados, divisao['feature'], divisao['valor'])
    
    arvore = {'feature': divisao['feature'], 'valor': divisao['valor']}
    arvore['esquerda'] = construir_arvore_recursivo(grupo_esquerda, profundidade_max, profundidade_atual + 1)
    arvore['direita'] = construir_arvore_recursivo(grupo_direita, profundidade_max, profundidade_atual + 1)
    
    return arvore

def construir_arvore_id3(treino, profundidade_max=10):
    return construir_arvore_recursivo(treino, profundidade_max, 0)

def predizer_amostra_arvore(arvore, amostra):
    """Navega na árvore para classificar uma única amostra."""
    if isinstance(arvore, (int, float)): # Se for um nó folha
        return arvore
    
    feature_idx, valor_divisao = arvore['feature'], arvore['valor']
    
    if amostra[feature_idx] < valor_divisao:
        return predizer_amostra_arvore(arvore['esquerda'], amostra)
    else:
        return predizer_amostra_arvore(arvore['direita'], amostra)


if __name__ == '__main__':
    caminho_arquivo = '../files/treino_sinais_vitais_com_label.txt'
    dataset_completo = carregar_dados(caminho_arquivo)
    num_features = len(dataset_completo[0]) - 1
    min_max = [[min(row[i] for row in dataset_completo), max(row[i] for row in dataset_completo)] for i in range(num_features)]
    
    for row in dataset_completo:
        for i in range(num_features):
            row[i] = (row[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])

    treino, teste = dividir_treino_teste(dataset_completo, 0.01)

    # --- Teste da Árvore de Decisão ---
    print("--- Treinando Árvore de Decisão (ID3) ---")
    arvore = construir_arvore_id3(treino, profundidade_max=5)
    predicoes_arvore = [predizer_amostra_arvore(arvore, linha[:-1]) for linha in teste]
    acuracia_arvore = calcular_acuracia(teste, predicoes_arvore)
    print(f"Acurácia da Árvore de Decisão: {acuracia_arvore:.2f}%\n")
