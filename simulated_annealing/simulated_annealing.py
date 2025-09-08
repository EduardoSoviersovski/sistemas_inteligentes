import math
import random
import time

import numpy as np

from matplotlib import pyplot as plt

from commons.utils import (
    export_results_to_csv,
    plot_results_scatterplot,
    route_length,
    two_opt_swap,
    Route,
    load_points_from_csv,
    distance_matrix,
    plot_route,
    create_first_route,
)

TMIN = 0.001
# ALPHA = 0.98
# LOG_ALPHA = 1.0
FILE = "files/20.csv"
PLOT = True


def simulated_annealing(
    dist_matrix: np.ndarray, configurations: dict, schedule: str
) -> tuple[Route, float]:
    rng = random.Random()
    n = len(dist_matrix)

    current = create_first_route(dist_matrix, start=rng.randrange(n))
    current_cost = route_length(current, dist_matrix)
    best, best_cost = current[:], current_cost

    T0 = configurations["T0"]
    ITERS_PER_T = configurations["ITERS_PER_T"]
    ALPHA = configurations["ALPHA"]
    LOG_ALPHA = configurations["LOG_ALPHA"]
    BETA = configurations["BETA"]

    T = T0
    iteration = 1
    while T > TMIN:
        for _ in range(ITERS_PER_T):
            i, k = sorted(rng.sample(range(n), 2))
            candidate = two_opt_swap(current, i, k)
            cand_cost = route_length(candidate, dist_matrix)
            delta = cand_cost - current_cost
            if delta < 0 or (rng.random() < math.exp(-delta / T)):
                current, current_cost = candidate, cand_cost
                if current_cost < best_cost:
                    best, best_cost = current[:], current_cost
        T = schedule_handler(
            T0, T, ITERS_PER_T, iteration, ALPHA, LOG_ALPHA, BETA, schedule
        )
        iteration += 1
    return best, best_cost


def schedule_handler(
    t0: float,
    t: float,
    iters_per_t: int,
    iteration: int,
    alpha: float,
    log_alpha: float,
    beta: float,
    schedule: str,
) -> float:
    if schedule == "geometric":
        t *= alpha
    elif schedule == "linear":
        t -= beta
    elif schedule == "logarithmic":
        t = t / (1 + log_alpha * math.log(1 + iteration))
    return t


def main():
    pts = load_points_from_csv(FILE)
    dist_matrix = distance_matrix(pts)
    n = len(dist_matrix)
    T0 = math.sqrt(n)
    ITERS_PER_T = n * 5
    configurations = [
        {
            "name": "1: Rápida e Agressiva",
            "ALPHA": 0.80,
            "BETA": 1.0,
            "LOG_ALPHA": 2.0,
            "T0": T0,
            "ITERS_PER_T": ITERS_PER_T,
        },
        {
            "name": "2: Clássica e Balanceada",
            "ALPHA": 0.90,
            "BETA": (T0 - TMIN) / 100,
            "LOG_ALPHA": 1.0,
            "T0": T0,
            "ITERS_PER_T": ITERS_PER_T,
        },
        {
            "name": "3: Lenta e Exaustiva",
            "ALPHA": 0.99,
            "BETA": (T0 - TMIN) / ITERS_PER_T,
            "LOG_ALPHA": 0.8,
            "T0": T0,
            "ITERS_PER_T": ITERS_PER_T,
        },
    ]

    color_map = {
        "1: Rápida e Agressiva": "#9400D3",  # Roxo Escuro
        "2: Clássica e Balanceada": "#FF8C00",  # Laranja Escuro
        "3: Lenta e Exaustiva": "#008080",  # Teal
    }

    # 1. Lista ÚNICA para guardar TODOS os resultados
    all_run_results = []

    cooling_schedule = ["linear", "geometric", "logarithmic"]
    # cooling_schedule = ["logarithmic"]

    # --- LAÇOS DE EXECUÇÃO ---
    for schedule in cooling_schedule:
        for configuration in configurations:
            print(
                f"Executando: Schedule='{schedule}', Config='{configuration['name']}'..."
            )
            for i in range(200):
                start_time = time.time()
                route, length = simulated_annealing(
                    dist_matrix, configuration, schedule
                )
                end_time = time.time()
                execution_time = end_time - start_time

                all_run_results.append(
                    {
                        "route": route,
                        "length": length,
                        "time": execution_time,
                        "schedule": schedule,
                        "config_name": configuration["name"],
                    }
                )
                print(
                    f"  -> Execução {i+1:02d}: Comprimento = {length:.4f}, Tempo = {execution_time:.4f}s"
                )

    # --- ANÁLISE E PLOTAGEM GERAL (APÓS TODAS AS EXECUÇÕES) ---
    if not all_run_results:
        print("Nenhum resultado para analisar.")
        return

    # 3. Encontra o melhor e o pior resultado GERAL de toda a lista
    best_overall_result = min(all_run_results, key=lambda r: r["length"])
    worst_overall_result = max(all_run_results, key=lambda r: r["length"])

    print("\n\n" + "=" * 80)
    print("RESUMO GERAL DE TODAS AS EXECUÇÕES")
    print("=" * 80)

    print("\nMELHOR RESULTADO ENCONTRADO:")
    print(f"   Comprimento: {best_overall_result['length']:.4f}")
    print(f"   Tempo:         {best_overall_result['time']:.4f}s")
    print(f"   Schedule:      '{best_overall_result['schedule']}'")
    print(f"   Configuração:  '{best_overall_result['config_name']}'")

    print("\nPIOR RESULTADO ENCONTRADO:")
    print(f"   Comprimento: {worst_overall_result['length']:.4f}")
    print(f"   Tempo:         {worst_overall_result['time']:.4f}s")
    print(f"   Schedule:      '{worst_overall_result['schedule']}'")
    print(f"   Configuração:  '{worst_overall_result['config_name']}'")
    print("=" * 80)

    export_results_to_csv(all_run_results)

    if PLOT:
        # Plota a melhor e a pior rota geral
        print("\nGerando gráficos da melhor e pior rota geral...")

        # --- PLOT DA MELHOR ROTA GERAL ---
        best_title = (
            f"Melhor Rota Geral (Comprimento: {best_overall_result['length']:.2f})\n"
            f"Schedule: {best_overall_result['schedule']} | Config: {best_overall_result['config_name']}"
        )
        plot_route(pts, best_overall_result["route"], best_title)

        # --- PLOT DA PIOR ROTA GERAL ---
        worst_title = (
            f"Pior Rota Geral (Comprimento: {worst_overall_result['length']:.2f})\n"
            f"Schedule: {worst_overall_result['schedule']} | Config: {worst_overall_result['config_name']}"
        )
        plot_route(pts, worst_overall_result["route"], worst_title)

        # Loop para criar um GRÁFICO DE DISPERSÃO por schedule
        print("\nGerando gráficos de dispersão por schedule...")
        for schedule in cooling_schedule:
            # Filtra os resultados para o schedule atual
            results_for_schedule = [
                r for r in all_run_results if r["schedule"] == schedule
            ]

            # Cria o título para o gráfico
            scatter_title = (
                f"Resultados das Configurações (Schedule: {schedule.capitalize()})"
            )

            # Chama a nova e simples função de plotagem
            plot_results_scatterplot(results_for_schedule, scatter_title, color_map)

        print("\nExibindo todos os gráficos...")
        plt.show()


if __name__ == "__main__":
    main()
