import math
import random
import time

import numpy as np

from matplotlib import pyplot as plt

from commons.utils import (
    export_results_to_csv,
    route_length,
    two_opt_swap,
    Route,
    load_points_from_csv,
    distance_matrix,
    create_first_route, get_best_and_worst_results, print_overall_results, plot_results,
)

TMIN = 0.001
FILE = "files/50.csv"
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
            T, iteration, ALPHA, LOG_ALPHA, BETA, schedule
        )
        iteration += 1
    return best, best_cost


def schedule_handler(
    t: float,
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

    all_run_results = []

    cooling_schedule = ["linear", "geometric", "logarithmic"]
    for schedule in cooling_schedule:
        for configuration in configurations:
            print(
                f"Executando: Schedule='{schedule}', Config='{configuration['name']}'..."
            )
            for i in range(5):
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

    if not all_run_results:
        print("Nenhum resultado para analisar.")
        return

    best_overall_result, worst_overall_result = get_best_and_worst_results(all_run_results)

    print_overall_results(best_overall_result, worst_overall_result, "schedule")

    export_results_to_csv(all_run_results, "simulated_annealing")
    if PLOT:
        plot_results(
            pts,
            best_overall_result,
            worst_overall_result,
            all_run_results,
            cooling_schedule,
            "schedule",
            color_map
        )
        plt.show()

if __name__ == "__main__":
    main()
