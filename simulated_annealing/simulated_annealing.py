import math
import random
import time

import numpy as np

from matplotlib import pyplot as plt

from commons.utils import (route_length, two_opt_swap, Route, load_points_from_csv,
   distance_matrix, plot_route, create_first_route)

TMIN = 0.001
ALPHA = 0.98
LOG_ALPHA = 1.0
FILE = "files/50.csv"
PLOT = True

COOLING_SCHEDULES = "linear"  # geometric | linear | logarithmic

def simulated_annealing(
    dist_matrix: np.ndarray,
) -> tuple[Route, float]:
    rng = random.Random()

    n = len(dist_matrix)
    T0 = math.sqrt(n)
    ITERS_PER_T = n * 5

    current = create_first_route(dist_matrix, start=rng.randrange(n))
    current_cost = route_length(current, dist_matrix)
    best, best_cost = current[:], current_cost

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
        T = schedule_handler(T0, T, ITERS_PER_T, iteration)
        iteration += 1
    return best, best_cost

def schedule_handler(t0: float, t: float, iters_per_t:int, iteration: int) -> float:
    if COOLING_SCHEDULES == "geometric":
        t *= ALPHA
    elif COOLING_SCHEDULES == "linear":
        # beta = t0 / iters_per_t TODO: COLCOAR NO RELATORIO QUE NAO É PRA USAR PQ DEMORA MUITO
        beta = 1.0
        t -= beta
    elif COOLING_SCHEDULES == "logarithmic":
        t = t / (1 + LOG_ALPHA * math.log(1 + iteration))
    return t

def main():
    pts = load_points_from_csv(FILE)
    dist_matrix = distance_matrix(pts)

    results_and_lengths = []
    for _ in range(100):
        start_time = time.time()
        route, length = simulated_annealing(dist_matrix)
        print(f"[SA] comprimento da rota: {length:.4f}")
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execução: [SA] comprimento = {length:.4f}, tempo = {execution_time:.4f}s")
        results_and_lengths.append({
            "route": route,
            "length": length,
            "time": execution_time
        })
    if PLOT:
        lengths = [r["length"] for r in results_and_lengths]
        times = [r["time"] for r in results_and_lengths]

        best_dist_index = lengths.index(min(lengths))
        worst_dist_index = lengths.index(max(lengths))
        average_length = sum(lengths) / len(lengths)

        min_time = min(times)
        max_time = max(times)
        average_time = sum(times) / len(times)

        print("\nResumo dos resultados:")
        print("---------------------------------")
        print(f"Melhor distância: {lengths[best_dist_index]:.4f}")
        print(f"Pior distância:   {lengths[worst_dist_index]:.4f}")
        print(f"Distância média:  {average_length:.4f}")
        print("---------------------------------")
        print(f"Melhor tempo: {min_time:.4f}s")
        print(f"Pior tempo:   {max_time:.4f}s")
        print(f"Tempo médio:  {average_time:.4f}s")
        print("---------------------------------")

        print(f"Melhor rota: {results_and_lengths[best_dist_index]['route']}")

        plot_route(pts, results_and_lengths[best_dist_index]['route'],
                   f"SA – Melhor Rota (Distância: {lengths[best_dist_index]:.4f})", show_ids=True)
        plot_route(pts, results_and_lengths[worst_dist_index]['route'],
                   f"SA – Pior Rota (Distância: {lengths[worst_dist_index]:.4f})", show_ids=True)

        plt.show()

if __name__ == '__main__':
    main()