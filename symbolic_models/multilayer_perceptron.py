import math
import random

from symbolic_models.utils import carregar_dados, dividir_treino_teste, calcular_acuracia

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

    treino_df, teste_df = dividir_treino_teste(dataset_completo, 0.3)

    print("--- Treinando Rede Neural ---")

    treino_list = treino_df.values.tolist()
    teste_list = teste_df.values.tolist()

    n_entradas = len(features_cols)

    labels_unicos = dataset_completo[nome_label].unique()
    n_saidas = len(labels_unicos)
    n_ocultas = 10

    print(f"Entradas: {n_entradas}, Saídas: {n_saidas} (Classes: {sorted(labels_unicos)})")

    nn = RedeNeural(n_entradas, n_ocultas, n_saidas)

    nn.treinar(treino_list, epocas=1000, taxa_aprendizado=0.3)

    predicoes_nn = [nn.predizer(linha[:-1]) for linha in teste_list]

    acuracia_nn = calcular_acuracia(teste_df, predicoes_nn)
    print(f"\nAcurácia da Rede Neural: {acuracia_nn:.2f}%\n")