import math
import random
import numpy as np

from symbolic_models.commons import random_oversample, visualizar_rede_neural
from symbolic_models.utils import dividir_treino_teste, carregar_dados, calcular_acuracia

CAMINHO_ARQUIVO = 'files/treino_sinais_vitais_com_label.txt'
NOME_LABEL = "label"
FEATURES_TO_IGNORE = ["index", "feature_6"]

TEST_PROPORTION = 0.2
VALIDATION_PROPORTION = 0.2

# --- Hiperparâmetros Melhorados ---
TAXA_APRENDIZADO = 0.01
N_OCULTAS = 7
EPOCHS = 1000
BATCH_SIZE = 16
MOMENTUM_BETA = 0.9
L2_LAMBDA = 1e-4

PACIENCIA_EARLY_STOP = 10  # Aumentada, pois o treino pode ser mais "barulhento"


# --- Funções de Ativação Vetorizadas (NumPy) ---

def relu(x):
    """Aplica a função ReLU elemento a elemento."""
    return np.maximum(0, x)


def relu_derivada(saida_relu):
    """Calcula a derivada da ReLU."""
    return (saida_relu > 0).astype(float)


def softmax(logits):
    """Calcula o Softmax de forma numericamente estável."""
    if logits.size == 0:
        return np.array([])

    # Previne overflow subtraindo o máximo
    max_logit = np.max(logits)
    exp_vals = np.exp(logits - max_logit)

    soma_exp_vals = np.sum(exp_vals)

    if soma_exp_vals == 0:
        return np.full_like(logits, 1.0 / logits.size)

    return exp_vals / soma_exp_vals


class RedeNeural:
    def __init__(self, n_entradas, n_ocultas, n_saidas):
        self.n_entradas = n_entradas
        self.n_ocultas = n_ocultas
        self.n_saidas = n_saidas

        # --- Inicialização de Pesos (He para ReLU) ---
        std_dev_oculta = math.sqrt(2.0 / self.n_entradas)
        std_dev_saida = math.sqrt(2.0 / self.n_ocultas)

        self.pesos_oculta = np.random.randn(self.n_entradas, self.n_ocultas) * std_dev_oculta
        self.bias_oculta = np.zeros(self.n_ocultas)
        self.pesos_saida = np.random.randn(self.n_ocultas, self.n_saidas) * std_dev_saida
        self.bias_saida = np.zeros(self.n_saidas)

        # --- Buffers para Momentum (Velocidade) ---
        self.v_pesos_oculta = np.zeros_like(self.pesos_oculta)
        self.v_bias_oculta = np.zeros_like(self.bias_oculta)
        self.v_pesos_saida = np.zeros_like(self.pesos_saida)
        self.v_bias_saida = np.zeros_like(self.bias_saida)

        # --- Modelos de "Melhor" para Early Stopping ---
        self.melhor_pesos_oculta = np.copy(self.pesos_oculta)
        self.melhor_bias_oculta = np.copy(self.bias_oculta)
        self.melhor_pesos_saida = np.copy(self.pesos_saida)
        self.melhor_bias_saida = np.copy(self.bias_saida)

        # --- Buffers para forward/backward pass ---
        self.saidas_oculta_atual = None
        self.ativacoes_oculta_raw = None
        self.saidas_final_atual = None

    def feedforward(self, entradas_np):
        """Executa o feedforward usando NumPy."""
        # Camada Oculta
        self.ativacoes_oculta_raw = np.dot(entradas_np, self.pesos_oculta) + self.bias_oculta
        self.saidas_oculta_atual = relu(self.ativacoes_oculta_raw)

        # Camada de Saída
        saidas_logits = np.dot(self.saidas_oculta_atual, self.pesos_saida) + self.bias_saida
        self.saidas_final_atual = softmax(saidas_logits)

        return self.saidas_final_atual

    def calcular_gradientes(self, entradas_np, esperado_np):
        """Calcula os gradientes para uma única amostra."""

        # 1. Calcular Deltas (Erros)
        # O gradiente do Cross-Entropy com Softmax é simplesmente (probabilidade - esperado)
        erros_saida = self.saidas_final_atual - esperado_np

        # Erro da camada oculta (backpropagado)
        derivada_relu = relu_derivada(self.saidas_oculta_atual)
        erros_oculta = np.dot(erros_saida, self.pesos_saida.T) * derivada_relu

        # 2. Calcular Gradientes dos Pesos/Bias
        # np.outer(A, B) é equivalente a A (vetor coluna) * B (vetor linha)
        grad_pesos_saida = np.outer(self.saidas_oculta_atual, erros_saida)
        grad_bias_saida = erros_saida  # O gradiente do bias é apenas o delta

        grad_pesos_oculta = np.outer(entradas_np, erros_oculta)
        grad_bias_oculta = erros_oculta

        return grad_pesos_oculta, grad_bias_oculta, grad_pesos_saida, grad_bias_saida

    def _calcular_loss_validacao(self, dados_validacao):
        """Calcula o Cross-Entropy Loss médio no conjunto de validação."""
        if not dados_validacao:
            return 0

        soma_erro_validacao = 0.0
        validos_contados = 0

        for linha in dados_validacao:
            entradas = np.array(linha[:-1])
            try:
                label_idx = int(linha[-1]) - 1
                if not (0 <= label_idx < self.n_saidas):
                    continue
            except (ValueError, IndexError):
                continue

            validos_contados += 1
            saidas_probs = self.feedforward(entradas)

            # Cross-Entropy Loss
            prob_correta = saidas_probs[label_idx]
            safe_prob = np.maximum(prob_correta, 1e-12)
            soma_erro_validacao += -math.log(safe_prob)

        return soma_erro_validacao / validos_contados if validos_contados > 0 else 0

    def _salvar_melhor_modelo(self):
        """Salva uma cópia dos pesos atuais."""
        self.melhor_pesos_oculta = np.copy(self.pesos_oculta)
        self.melhor_bias_oculta = np.copy(self.bias_oculta)
        self.melhor_pesos_saida = np.copy(self.pesos_saida)
        self.melhor_bias_saida = np.copy(self.bias_saida)

    def _restaurar_melhor_modelo(self):
        """Restaura os pesos do melhor modelo salvo."""
        print("Restaurando pesos do melhor modelo encontrado...")
        self.pesos_oculta = np.copy(self.melhor_pesos_oculta)
        self.bias_oculta = np.copy(self.melhor_bias_oculta)
        self.pesos_saida = np.copy(self.melhor_pesos_saida)
        self.bias_saida = np.copy(self.melhor_bias_saida)

    def treinar(self, dados_treino, dados_validacao):

        n_amostras = len(dados_treino)
        if n_amostras == 0:
            print("Erro: dados_treino está vazio.")
            return

        melhor_erro_validacao = float('inf')
        epocas_sem_melhora = 0

        print(f"Iniciando treino com {n_amostras} amostras, Batch Size={BATCH_SIZE}, "
              f"LR={TAXA_APRENDIZADO}, Momentum={MOMENTUM_BETA}, L2={L2_LAMBDA}")

        for epoca in range(EPOCHS):
            soma_erro_epoca = 0.0
            random.shuffle(dados_treino)

            # --- Loop de Mini-Batch ---
            for i in range(0, n_amostras, BATCH_SIZE):
                batch_data = dados_treino[i:i + BATCH_SIZE]
                if not batch_data:
                    continue

                n_batch_atual = len(batch_data)

                # Acumuladores de gradiente para o batch
                acum_grad_po = np.zeros_like(self.pesos_oculta)
                acum_grad_bo = np.zeros_like(self.bias_oculta)
                acum_grad_ps = np.zeros_like(self.pesos_saida)
                acum_grad_bs = np.zeros_like(self.bias_saida)

                # --- Loop dentro do Mini-Batch ---
                for linha in batch_data:
                    entradas_np = np.array(linha[:-1])
                    label_idx = int(linha[-1]) - 1

                    esperado_np = np.zeros(self.n_saidas)
                    if 0 <= label_idx < self.n_saidas:
                        esperado_np[label_idx] = 1.0

                    # 1. Forward pass
                    saidas_probs = self.feedforward(entradas_np)

                    # Acumula o loss da época
                    prob_correta = saidas_probs[label_idx]
                    safe_prob = np.maximum(prob_correta, 1e-12)
                    soma_erro_epoca += -math.log(safe_prob)

                    # 2. Backward pass (cálculo dos gradientes)
                    grad_po, grad_bo, grad_ps, grad_bs = self.calcular_gradientes(entradas_np, esperado_np)

                    # Acumula gradientes do batch
                    acum_grad_po += grad_po
                    acum_grad_bo += grad_bo
                    acum_grad_ps += grad_ps
                    acum_grad_bs += grad_bs

                # --- Fim do Mini-Batch: Atualização dos Pesos ---

                # 3. Média dos gradientes do batch
                grad_medio_po = acum_grad_po / n_batch_atual
                grad_medio_bo = acum_grad_bo / n_batch_atual
                grad_medio_ps = acum_grad_ps / n_batch_atual
                grad_medio_bs = acum_grad_bs / n_batch_atual

                # 4. Adicionar Regularização L2 (Weight Decay)
                # (Não se aplica ao bias)
                grad_medio_po += L2_LAMBDA * self.pesos_oculta
                grad_medio_ps += L2_LAMBDA * self.pesos_saida

                # 5. Aplicar Momentum
                self.v_pesos_oculta = (MOMENTUM_BETA * self.v_pesos_oculta) + grad_medio_po
                self.v_bias_oculta = (MOMENTUM_BETA * self.v_bias_oculta) + grad_medio_bo
                self.v_pesos_saida = (MOMENTUM_BETA * self.v_pesos_saida) + grad_medio_ps
                self.v_bias_saida = (MOMENTUM_BETA * self.v_bias_saida) + grad_medio_bs

                # 6. Atualizar Pesos e Biases
                self.pesos_oculta -= TAXA_APRENDIZADO * self.v_pesos_oculta
                self.bias_oculta -= TAXA_APRENDIZADO * self.v_bias_oculta
                self.pesos_saida -= TAXA_APRENDIZADO * self.v_pesos_saida
                self.bias_saida -= TAXA_APRENDIZADO * self.v_bias_saida

            # --- Fim da Época: Checagem de Early Stopping ---
            if epoca % 10 == 0:
                erro_treino_epoca = soma_erro_epoca / n_amostras
                erro_validacao_epoca = self._calcular_loss_validacao(dados_validacao)

                print(
                    f'> Época={epoca}, Erro Treino={erro_treino_epoca:.4f}, Erro Validação={erro_validacao_epoca:.4f}')

                if erro_validacao_epoca < melhor_erro_validacao:
                    melhor_erro_validacao = erro_validacao_epoca
                    epocas_sem_melhora = 0
                    self._salvar_melhor_modelo()
                    print(f'  -> Nova melhor pontuação de validação! Salvando modelo.')
                else:
                    epocas_sem_melhora += 1

                if epocas_sem_melhora >= PACIENCIA_EARLY_STOP:
                    print(f'\nPARADA ANTECIPADA! Sem melhora na validação por {PACIENCIA_EARLY_STOP} checagens.')
                    self._restaurar_melhor_modelo()
                    break

    def predizer(self, amostra):
        """Prediz a classe de uma única amostra."""
        # Garante que a entrada é um array numpy
        amostra_np = np.array(amostra)
        saidas_prob = self.feedforward(amostra_np)

        # np.argmax é muito mais eficiente
        return np.argmax(saidas_prob) + 1


if __name__ == '__main__':
    print("--- Carregando e Preparando Dados ---")

    dataset_completo = carregar_dados(
        CAMINHO_ARQUIVO,
        features_to_ignore=FEATURES_TO_IGNORE,
        label_col_name=NOME_LABEL
    )

    features_cols = [col for col in dataset_completo.columns if col != NOME_LABEL]

    # Normalização (Min-Max)
    min_vals = dataset_completo[features_cols].min()
    max_vals = dataset_completo[features_cols].max()
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Evita divisão por zero
    dataset_completo[features_cols] = (dataset_completo[features_cols] - min_vals) / range_vals

    # Divisão Treino/Teste/Validação
    treino_completo_df, teste_df = dividir_treino_teste(dataset_completo, TEST_PROPORTION)
    treino_nn_df, validacao_df = dividir_treino_teste(treino_completo_df, VALIDATION_PROPORTION)

    # Conversão para listas (como esperado pela sua rede)
    treino_list_original = treino_nn_df.values.tolist()
    validacao_list = validacao_df.values.tolist()
    teste_list = teste_df.values.tolist()

    print(f"Tam. Treino Original: {len(treino_list_original)}")
    print(f"Tam. Validação: {len(validacao_list)}")
    print(f"Tam. Teste Final: {len(teste_list)}")

    # Balanceamento
    treino_list_balanceada = random_oversample(treino_list_original)
    print(f"Tam. Treino Balanceado: {len(treino_list_balanceada)}")

    n_entradas = len(features_cols)
    labels_unicos = dataset_completo[NOME_LABEL].unique()
    labels_ordenados = sorted(labels_unicos)
    n_saidas = len(labels_unicos)

    print(f"Entradas: {n_entradas}, Ocultas: {N_OCULTAS}, Saídas: {n_saidas}")

    print("\n--- Treinando Rede Neural Otimizada ---")
    nn = RedeNeural(n_entradas, N_OCULTAS, n_saidas)

    # Chama o treino com os novos hiperparâmetros
    nn.treinar(
        dados_treino=treino_list_balanceada,
        dados_validacao=validacao_list
    )

    print("\n--- Avaliando Modelo ---")
    predicoes_nn = [nn.predizer(linha[:-1]) for linha in teste_list]
    acuracia_nn = calcular_acuracia(teste_df, predicoes_nn)

    try:
        visualizar_rede_neural(nn, features_cols, labels_ordenados, nome_arquivo='rede_neural_graph')
    except Exception as e:
        print(f"Erro ao gerar visualização da rede: {e}")

    print(f"\nAcurácia final da Rede Neural: {acuracia_nn:.2f}%\n")