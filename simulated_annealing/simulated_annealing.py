import math
import random
import numpy as np

from dataclasses import dataclass

from matplotlib import pyplot as plt

from commons.utils import (route_length, two_opt_swap, Route, load_points_from_csv,
   distance_matrix, plot_route, create_first_route)

T0 = 100.0
TMIN = 0.001
ALPHA = 0.98
ITERS_PER_T = 5
FILE = "files/50.csv"
PLOT = True


def simulated_annealing(
    dist_matrix: np.ndarray,
) -> tuple[Route, float]:
    rng = random.Random()
    n = len(dist_matrix)

    current = create_first_route(dist_matrix, start=rng.randrange(n))
    current_cost = route_length(current, dist_matrix)
    best, best_cost = current[:], current_cost

    T = T0
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
        T *= ALPHA
    return best, best_cost


def main():
    pts = load_points_from_csv(FILE)
    dist_matrix = distance_matrix(pts)

    results_and_lengths = []
    for _ in range(10):
        route, length = simulated_annealing(dist_matrix)
        print(f"[SA] comprimento da rota: {length:.4f}")
        results_and_lengths.append({"route":route, "length": length})
    if PLOT:
        lengths = [r["length"] for r in results_and_lengths]
        best_index = lengths.index(min(lengths))
        worst_index = lengths.index(max(lengths))
        average_length = sum(lengths) / len(lengths)

        print("\nResumo dos resultados:")
        print(f"Melhor rota: {results_and_lengths[best_index]['route']}")
        print(f"Melhor distância: {lengths[best_index]:.4f}")
        print(f"Pior rota: {results_and_lengths[worst_index]['route']}")
        print(f"Pior distância: {lengths[worst_index]:.4f}")
        print(f"Distância média: {average_length:.4f}")
        plot_route(pts, results_and_lengths[best_index]['route'],
                   f"SA – melhor rota {lengths[best_index]:.4f}", show_ids=True)

        plot_route(pts, results_and_lengths[worst_index]['route'],
                   f"SA – pior rota {lengths[worst_index]:.4f}", show_ids=True)

        plt.show()

if __name__ == '__main__':
    main()