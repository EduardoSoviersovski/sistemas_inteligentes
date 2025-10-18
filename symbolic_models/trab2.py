import math
import random
from collections import Counter

# --- 1. Carregamento e Preparo dos Dados ---

def carregar_dados(caminho_arquivo):
    """Lê o arquivo de dados e o converte para uma lista de listas."""
    dataset = []
    with open(caminho_arquivo, 'r') as f:
        for linha in f:
            if not linha.strip():
                continue
            # Converte cada valor para float, exceto o ID que é ignorado
            partes = linha.strip().split(',')
            # ID (partes[0]) é ignorado, features são convertidas para float, label para int
            features = [float(x) for x in partes[1:-1]]
            label = int(partes[-1])
            dataset.append(features + [label])
    return dataset

def dividir_treino_teste(dataset, proporcao_teste=0.2):
    """Divide o dataset em conjuntos de treino e teste."""
    random.shuffle(dataset)
    ponto_divisao = int(len(dataset) * (1 - proporcao_teste))
    treino = dataset[:ponto_divisao]
    teste = dataset[ponto_divisao:]
    return treino, teste

# --- 2. Algoritmo 1: Árvore de Decisão (ID3) ---

def calcular_entropia(dados):
    """Calcula a entropia de um conjunto de dados."""
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
    """Divide os dados com base em um valor de uma feature."""
    grupo_esquerda = [linha for linha in dados if linha[indice_feature] < valor]
    grupo_direita = [linha for linha in dados if linha[indice_feature] >= valor]
    return grupo_esquerda, grupo_direita

def encontrar_melhor_divisao(dados):
    """Encontra a melhor feature e valor para dividir os dados."""
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
    """Retorna a classe mais comum em um grupo (nó folha)."""
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

# --- 3. Algoritmo 2: Random Forest ---

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

# --- 4. Algoritmo 3: Rede Neural (MLP) ---

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivada(x):
    return x * (1 - x)

class RedeNeural:
    def __init__(self, n_entradas, n_ocultas, n_saidas):
        self.n_entradas = n_entradas
        self.n_ocultas = n_ocultas
        self.n_saidas = n_saidas
        
        # Inicializa pesos e biases aleatoriamente
        self.pesos_oculta = [[random.uniform(-0.5, 0.5) for _ in range(self.n_ocultas)] for _ in range(self.n_entradas)]
        self.bias_oculta = [random.uniform(-0.5, 0.5) for _ in range(self.n_ocultas)]
        
        self.pesos_saida = [[random.uniform(-0.5, 0.5) for _ in range(self.n_saidas)] for _ in range(self.n_ocultas)]
        self.bias_saida = [random.uniform(-0.5, 0.5) for _ in range(self.n_saidas)]

    def feedforward(self, entradas):
        # Camada oculta
        saidas_oculta = [0] * self.n_ocultas
        for j in range(self.n_ocultas):
            ativacao = self.bias_oculta[j]
            for i in range(self.n_entradas):
                ativacao += entradas[i] * self.pesos_oculta[i][j]
            saidas_oculta[j] = sigmoid(ativacao)
        self.saidas_oculta_atual = saidas_oculta
        
        # Camada de saída
        saidas_final = [0] * self.n_saidas
        for j in range(self.n_saidas):
            ativacao = self.bias_saida[j]
            for i in range(self.n_ocultas):
                ativacao += saidas_oculta[i] * self.pesos_saida[i][j]
            saidas_final[j] = sigmoid(ativacao)
        self.saidas_final_atual = saidas_final
        
        return saidas_final

    def backpropagation(self, entradas, esperado, taxa_aprendizado):
        # Erro na camada de saída
        erros_saida = [0] * self.n_saidas
        for i in range(self.n_saidas):
            erros_saida[i] = (esperado[i] - self.saidas_final_atual[i]) * sigmoid_derivada(self.saidas_final_atual[i])

        # Erro na camada oculta
        erros_oculta = [0] * self.n_ocultas
        for i in range(self.n_ocultas):
            erro = 0.0
            for j in range(self.n_saidas):
                erro += erros_saida[j] * self.pesos_saida[i][j]
            erros_oculta[i] = erro * sigmoid_derivada(self.saidas_oculta_atual[i])

        # Atualiza pesos e biases da camada de saída
        for i in range(self.n_ocultas):
            for j in range(self.n_saidas):
                self.pesos_saida[i][j] += taxa_aprendizado * erros_saida[j] * self.saidas_oculta_atual[i]
        for i in range(self.n_saidas):
            self.bias_saida[i] += taxa_aprendizado * erros_saida[i]

        # Atualiza pesos e biases da camada oculta
        for i in range(self.n_entradas):
            for j in range(self.n_ocultas):
                self.pesos_oculta[i][j] += taxa_aprendizado * erros_oculta[j] * entradas[i]
        for i in range(self.n_ocultas):
            self.bias_oculta[i] += taxa_aprendizado * erros_oculta[i]

    def treinar(self, dados_treino, epocas, taxa_aprendizado):
        for epoca in range(epocas):
            soma_erro = 0
            for linha in dados_treino:
                entradas = linha[:-1]
                
                # One-hot encoding para o label esperado
                esperado = [0] * self.n_saidas
                # Labels são de 1 a 4, então o índice é label-1
                label_idx = int(linha[-1]) - 1
                esperado[label_idx] = 1
                
                saidas = self.feedforward(entradas)
                soma_erro += sum([(esperado[i] - saidas[i])**2 for i in range(self.n_saidas)])
                
                self.backpropagation(entradas, esperado, taxa_aprendizado)
            if (epoca % 100) == 0:
                print(f'> época={epoca}, erro={soma_erro:.4f}')

    def predizer(self, amostra):
        saidas = self.feedforward(amostra)
        # Retorna o índice da maior saída + 1 (pois os labels são de 1 a 4)
        return saidas.index(max(saidas)) + 1


# --- 5. Avaliação ---

def calcular_acuracia(dados_reais, predicoes):
    corretos = 0
    for i in range(len(dados_reais)):
        if dados_reais[i][-1] == predicoes[i]:
            corretos += 1
    return corretos / float(len(dados_reais)) * 100.0


# --- Execução Principal ---

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

    # --- Teste da Árvore de Decisão ---
    print("--- Treinando Árvore de Decisão (ID3) ---")
    arvore = construir_arvore_id3(treino, profundidade_max=5)
    predicoes_arvore = [predizer_amostra_arvore(arvore, linha[:-1]) for linha in teste]
    acuracia_arvore = calcular_acuracia(teste, predicoes_arvore)
    print(f"Acurácia da Árvore de Decisão: {acuracia_arvore:.2f}%\n")

    # --- Teste do Random Forest ---
    print("--- Treinando Random Forest ---")
    floresta = random_forest_treino(treino, n_arvores=10, profundidade_max=5)
    predicoes_rf = [random_forest_predicao(floresta, linha[:-1]) for linha in teste]
    acuracia_rf = calcular_acuracia(teste, predicoes_rf)
    print(f"Acurácia do Random Forest: {acuracia_rf:.2f}%\n")

    # --- Teste da Rede Neural ---
    print("--- Treinando Rede Neural ---")
    n_entradas = len(treino[0]) - 1
    n_saidas = len(set(row[-1] for row in treino)) # 4 classes
    n_ocultas = 10 # Número de neurônios na camada oculta
    
    nn = RedeNeural(n_entradas, n_ocultas, n_saidas)
    nn.treinar(treino, epocas=1000, taxa_aprendizado=0.3)
    
    predicoes_nn = [nn.predizer(linha[:-1]) for linha in teste]
    acuracia_nn = calcular_acuracia(teste, predicoes_nn)
    print(f"\nAcurácia da Rede Neural: {acuracia_nn:.2f}%\n")