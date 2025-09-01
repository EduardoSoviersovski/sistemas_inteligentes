import argparse
import math
import random
import numpy as np

from dataclasses import dataclass

from matplotlib import pyplot as plt

from commons.utils import (route_length, two_opt_swap, Route, load_points_from_csv,
   random_points, distance_matrix, plot_route, create_first_route)

T0 = 100.0
Tmin = 0.001
alpha = 0.98
iters_per_T = 5


@dataclass
class SAParams:
    T0: float = 100.0
    Tmin: float = 0.001
    alpha: float = 0.98
    iters_per_T: int = 5


def simulated_annealing(
    dist_matrix: np.ndarray,
    seed: int = None,
    params: SAParams = SAParams()
) -> tuple[Route, float]:
    rng = random.Random()
    n = len(dist_matrix)

    current = create_first_route(dist_matrix, start=rng.randrange(n))
    current_cost = route_length(current, dist_matrix)
    best, best_cost = current[:], current_cost


    T = params.T0
    while T > params.Tmin:
        for _ in range(params.iters_per_T):
            i, k = sorted(rng.sample(range(n), 2))
            candidate = two_opt_swap(current, i, k)
            cand_cost = route_length(candidate, dist_matrix)
            delta = cand_cost - current_cost
            if delta < 0 or (rng.random() < math.exp(-delta / T)):
                current, current_cost = candidate, cand_cost
                if current_cost < best_cost:
                    best, best_cost = current[:], current_cost
        T *= params.alpha
    return best, best_cost


def main():
    parser = argparse.ArgumentParser(description="TSP – SA")
    parser.add_argument('--n', type=int, default=50, help='número de cidades (se não usar --file)')
    parser.add_argument('--file', type=str, default=None, help='CSV com colunas id,x,y OU x,y')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--plot', action='store_true', help='plota as melhores rotas')

    parser.add_argument('--T0', type=float, default=T0)
    parser.add_argument('--Tmin', type=float, default=Tmin)
    parser.add_argument('--alpha', type=float, default=alpha)
    parser.add_argument('--iters-per-T', type=int, default=iters_per_T)

    args = parser.parse_args()
    rng = random.Random(args.seed)

    if args.file:
        pts = load_points_from_csv(args.file)
    else:
        pts = random_points(args.n, rng)

    dist_matrix = distance_matrix(pts)

    params = SAParams(T0=args.T0, Tmin=args.Tmin, alpha=args.alpha, iters_per_T=args.iters_per_T)

    results_and_lengths = []
    for _ in range(10):
        route, length = simulated_annealing(dist_matrix, seed=args.seed, params=params)
        print(f"[SA] comprimento da rota: {length:.4f}")
        results_and_lengths.append({"route":route, "length": length})
    if args.plot:
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