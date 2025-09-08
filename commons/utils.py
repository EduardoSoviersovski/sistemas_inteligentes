from collections import defaultdict
import csv
from datetime import datetime
import math
import random
from typing import List

import numpy as np
import matplotlib.pyplot as plt

Point = tuple[float, float, str]
Route = list[int]

def euclidean(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def distance_matrix(points: list[Point]) -> np.ndarray:
    n = len(points)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean(points[i], points[j])
            D[i, j] = D[j, i] = d
    return D


def route_length(route: Route, D: np.ndarray) -> float:
    n = len(route)
    total = 0.0
    for i in range(n):
        total += D[route[i], route[(i + 1) % n]]
    return total


def create_random_route(cities: list[int], rng: random.Random) -> Route:
    route = list(cities)
    rng.shuffle(route[1:])
    return route

def create_first_route(D: np.ndarray, start: int = 0) -> Route:
    n = len(D)
    unvisited = set(range(n))
    route = [start]
    unvisited.remove(start)
    while unvisited:
        last = route[-1]
        nxt = min(unvisited, key=lambda j: D[last, j])
        route.append(nxt)
        unvisited.remove(nxt)
    return route


def two_opt_swap(route: Route, i: int, k: int) -> Route:
    return route[:i] + list(reversed(route[i:k + 1])) + route[k + 1:]


def load_points_from_csv(path: str) -> list[Point]:
    pts: list[Point] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(';', ',').split(',')
            if len(parts) >= 3 and parts[0].strip().lower() in {"id", "index"}:
                continue
            if len(parts) == 3:
                name, x, y = parts
            else:
                name = str(len(pts))
                x, y = parts[:2]
            pts.append((float(x), float(y), name))
    return pts


def export_results_to_csv(all_results: list):
    """
    Exporta os resultados completos e um resumo agregado para arquivos CSV,
    usando um timestamp no nome do arquivo para garantir unicidade.
    """
    if not all_results:
        print("Nenhum resultado para exportar.")
        return

    # --- NOVO: Gera um timestamp para usar no nome do arquivo ---
    # Pega a data e hora atuais
    now = datetime.now()
    # Formata como 'AAAA-MM-DD_HH-MM-SS' (ex: '2025-09-07_22-27-52')
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    # -----------------------------------------------------------

    print(f"\nExportando resultados para arquivos CSV com o timestamp '{timestamp}'...")

    # --- GERAÇÃO DO CSV DETALHADO ---
    detailed_headers = list(all_results[0].keys())
    detailed_headers.remove('route')
    detailed_headers.append('route_str')
    if 'run_id' not in detailed_headers:
        detailed_headers.insert(0, 'run_id')

    # NOVO: Usa o timestamp no nome do arquivo
    detailed_filename = f"resultados_completos_{timestamp}.csv"
    with open(detailed_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=detailed_headers)
        writer.writeheader()
        for i, res in enumerate(all_results):
            row = res.copy()
            row['run_id'] = i + 1
            row['route_str'] = '-'.join(map(str, row.pop('route')))
            for key in list(row.keys()):
                if not isinstance(row[key], (int, float, str, bool)):
                    del row[key]
            writer.writerow(row)
    
    print(f" -> '{detailed_filename}' gerado com sucesso.")

    # --- GERAÇÃO DO CSV RESUMIDO ---
    summary_headers = [
        'schedule', 'config_name', 'num_execucoes',
        'comprimento_medio', 'comprimento_minimo', 'comprimento_maximo', 'comprimento_desvio_padrao',
        'tempo_medio_s', 'tempo_minimo_s', 'tempo_maximo_s'
    ]
    
    grouped = defaultdict(list)
    for res in all_results:
        grouped[(res['schedule'], res['config_name'])].append(res)

    summary_data = []
    for (schedule, config), results in grouped.items():
        lengths = [r['length'] for r in results]
        times = [r['time'] for r in results]
        summary_data.append({
            'schedule': schedule,
            'config_name': config,
            'num_execucoes': len(results),
            'comprimento_medio': np.mean(lengths),
            'comprimento_minimo': np.min(lengths),
            'comprimento_maximo': np.max(lengths),
            'comprimento_desvio_padrao': np.std(lengths),
            'tempo_medio_s': np.mean(times),
            'tempo_minimo_s': np.min(times),
            'tempo_maximo_s': np.max(times)
        })

    # NOVO: Usa o mesmo timestamp no nome do segundo arquivo
    summary_filename = f"resumo_resultados_{timestamp}.csv"
    with open(summary_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=summary_headers)
        writer.writeheader()
        writer.writerows(summary_data)

    print(f" -> '{summary_filename}' gerado com sucesso.")

def plot_route(points: List[Point], route: Route, title: str, show_ids=False):
    """
    Plota os pontos e a rota que os conecta.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plota a rota conectando os pontos
    x_coords = [points[i][0] for i in route]
    y_coords = [points[i][1] for i in route]
    
    # Adiciona a linha de volta para o início para fechar o ciclo
    x_coords.append(points[route[0]][0])
    y_coords.append(points[route[0]][1])
    
    ax.plot(x_coords, y_coords, 'o-', markersize=5, linewidth=1.5, label='Rota')

    # Plota o ponto inicial com destaque
    ax.plot(x_coords[0], y_coords[0], 'o', markersize=10, color='red', label='Início/Fim')

    # Adiciona os IDs dos pontos se solicitado
    if show_ids:
        for i, (x, y) in enumerate(points):
            ax.text(x, y, str(i), fontsize=8, ha='center', va='bottom')

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Coordenada X")
    ax.set_ylabel("Coordenada Y")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()


def plot_results_scatterplot(results: list, title: str, color_map: dict):
    """
    Cria um gráfico de dispersão aprimorado para visualizar todos os resultados,
    com cores, marcadores, rótulos informativos e anotações.
    """
    if not results:
        print(f"Aviso: Não há dados para plotar o gráfico '{title}'")
        return

    fig, ax = plt.subplots(figsize=(14, 9))

    # Agrupa os resultados pela configuração
    grouped_results = defaultdict(list)
    for result in results:
        grouped_results[result['config_name']].append(result)

    # Plota os pontos de cada configuração
    for name, config_results in grouped_results.items():
        lengths = [r['length'] for r in config_results]
        times = [r['time'] for r in config_results]
        
        ax.scatter(lengths, times, 
                   label=name, 
                   color=color_map.get(name, 'gray'),
                #    marker=marker_map.get(name, 'x'),
                   alpha=0.8, 
                   s=80) # Tamanho dos pontos

    # Rótulos dos eixos mais informativos
    ax.set_xlabel('Comprimento da Rota', fontsize=12)
    ax.set_ylabel('Tempo de Execução (s)', fontsize=12)
    
    ax.set_title(title, fontsize=16, pad=20)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Legenda posicionada fora do gráfico para não obstruir os dados
    ax.legend(title='Configurações', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

    # Ajusta o layout para garantir que a legenda externa caiba na figura
    fig.tight_layout(rect=[0, 0, 0.85, 1])
