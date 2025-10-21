import random
from collections import Counter

from decision_tree import construir_arvore_id3, predizer_amostra_arvore
from utils import carregar_dados, dividir_treino_teste, calcular_acuracia

def criar_amostra_bootstrap(dataset):
    """Cria uma amostra do dataset com reposição."""
    amostra = [random.choice(dataset) for _ in range(len(dataset))]
    return amostra

def random_forest_treino(treino, n_arvores, profundidade_max):
    """Treina um modelo Random Forest."""
    floresta = []
    for _ in range(n_arvores):
        amostra_bootstrap = criar_amostra_bootstrap(treino)
        arvore = construir_arvore_id3(amostra_bootstrap, profundidade_max)
        floresta.append(arvore)
    return floresta

def random_forest_predicao(floresta, amostra):
    """Faz a predição baseada na votação majoritária da floresta."""
    predicoes = [predizer_amostra_arvore(arvore, amostra) for arvore in floresta]
    # Retorna o voto majoritário
    return Counter(predicoes).most_common(1)[0][0]

if __name__ == '__main__':
    caminho_arquivo = '../files/treino_sinais_vitais_com_label.txt'
    dataset_completo = carregar_dados(caminho_arquivo)
    
    # Normalização dos dados (importante para a Rede Neural)
    num_features = len(dataset_completo[0]) - 1
    min_max = [[min(row[i] for row in dataset_completo), max(row[i] for row in dataset_completo)] for i in range(num_features)]
    
    for row in dataset_completo:
        for i in range(num_features):
            row[i] = (row[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])

    treino, teste = dividir_treino_teste(dataset_completo, 0.01)

    # --- Teste do Random Forest ---
    print("--- Treinando Random Forest ---")
    floresta = random_forest_treino(treino, n_arvores=10, profundidade_max=5)
    predicoes_rf = [random_forest_predicao(floresta, linha[:-1]) for linha in teste]
    acuracia_rf = calcular_acuracia(teste, predicoes_rf)
    print(f"Acurácia do Random Forest: {acuracia_rf:.2f}%\n")
