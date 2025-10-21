import random

def carregar_dados(caminho_arquivo):
    dataset = []
    with open(caminho_arquivo, 'r') as f:
        for linha in f:
            if not linha.strip():
                continue
            partes = linha.strip().split(',')
            features = [float(x) for x in partes[1:-1]]
            label = int(partes[-1])
            dataset.append(features + [label])
    return dataset

def dividir_treino_teste(dataset, proporcao_teste=0.2):
    random.shuffle(dataset)
    ponto_divisao = int(len(dataset) * (1 - proporcao_teste))
    treino = dataset[:ponto_divisao]
    teste = dataset[ponto_divisao:]
    return treino, teste

def calcular_acuracia(dados_reais, predicoes):
    corretos = 0
    for i in range(len(dados_reais)):
        if dados_reais[i][-1] == predicoes[i]:
            corretos += 1
    return corretos / float(len(dados_reais)) * 100.0