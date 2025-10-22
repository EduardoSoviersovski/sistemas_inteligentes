import pandas as pd


def carregar_dados(caminho_arquivo, features_to_ignore=[], label_col_name=None):
    df = pd.read_csv(
        caminho_arquivo,
        header=0,  # A primeira linha é o cabeçalho
        skip_blank_lines=True
    )
    all_columns = df.columns.tolist()

    feature_col_names = [col for col in all_columns if col != label_col_name]

    if features_to_ignore is not None:
        ignore_set = set(features_to_ignore)
        feature_col_names = [col for col in feature_col_names if col not in ignore_set]

    df[label_col_name] = df[label_col_name].astype(int)

    colunas_finais = feature_col_names + [label_col_name]
    df_final = df[colunas_finais]

    return df_final

def dividir_treino_teste(dataset, proporcao_teste=0.2, seed=42):
    teste_df = dataset.sample(frac=proporcao_teste, random_state=seed)
    treino_df = dataset.drop(teste_df.index)
    return treino_df, teste_df

def calcular_acuracia(teste_df, predicoes):
    labels_reais = teste_df['label'].tolist()
    if len(labels_reais) != len(predicoes):
        print("Erro: Tamanho da lista de predições é diferente do teste.")
        return 0.0

    corretos = sum(1 for real, pred in zip(labels_reais, predicoes) if real == pred)
    return (corretos / len(labels_reais)) * 100.0


def imprimir_arvore(arvore, indentacao=""):
    """Imprime a árvore de decisão de forma legível no console."""

    # Caso base: é um nó folha (contém a predição da classe)
    if not isinstance(arvore, dict):
        print(f"{indentacao}-> Classe: {arvore}")
        return

    # É um nó de decisão
    feature = arvore['feature']
    valor = arvore['valor']
    print(f"{indentacao}[Feature {feature} < {valor:.4f} ?]")

    # Recursão para o ramo da esquerda (True)
    print(f"{indentacao}  ├─ True:")
    imprimir_arvore(arvore['esquerda'], indentacao + "  │   ")

    # Recursão para o ramo da direita (False)
    print(f"{indentacao}  └─ False:")
    imprimir_arvore(arvore['direita'], indentacao + "  │   ")