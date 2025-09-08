import math
import random
import time

import numpy as np
from matplotlib import pyplot as plt

from commons.utils import (
    route_length,
    load_points_from_csv,
    distance_matrix,
    two_opt_swap,
    Route, create_random_route, export_results_to_csv, get_best_and_worst_results,
    print_overall_results, plot_results
)
FILE = "files/50.csv"
PLOT = True

configurations = [
    {
        "name": "1: Clássica e Balanceada",
        "POPULATION_SIZE": 100,
        "MUTATION_RATE": 0.01,
        "TOURNAMENT_SIZE": 5,
        "GENERATIONS": 200,
    },
    {
        "name": "2: Rápida e Agressiva",
        "POPULATION_SIZE": 30,
        "MUTATION_RATE": 0.005,
        "TOURNAMENT_SIZE": 7,
        "GENERATIONS": 100,
    },
    {
        "name": "3: Lenta e Exaustiva",
        "POPULATION_SIZE": 300,
        "MUTATION_RATE": 0.05,
        "TOURNAMENT_SIZE": 3,
        "GENERATIONS": 500,
    },
    {
        "name": "4: Alta Mutação",
        "POPULATION_SIZE": 150,
        "MUTATION_RATE": 0.10,
        "TOURNAMENT_SIZE": 2,
        "GENERATIONS": 300,
    },
]

color_map = {
    "1: Clássica e Balanceada": "#9400D3",  # Roxo
    "2: Rápida e Agressiva": "#FF8C00",  # Laranja
    "3: Lenta e Exaustiva": "#008080",  # Teal
    "4: Alta Mutação": "#DC143C", # Vermelho
}

def initial_population(
        pop_size: int, dist_matrix: np.ndarray, rng: random.Random
) -> list[Route]:
    n_cities = len(dist_matrix)
    cities = list(range(n_cities))
    population = [create_random_route(cities, rng) for _ in range(pop_size)]
    return population


def rank_routes(population: list[Route], dist_matrix: np.ndarray) -> list[tuple[int, float]]:
    fitness_results = {}
    for i, route in enumerate(population):
        fitness_results[i] = 1 / route_length(route, dist_matrix)
    return sorted(fitness_results.items(), key=lambda x: x[1], reverse=True)

def selection(
        selection_method: str,
        ranked_population: list[tuple[int, float]],
        population: list[Route],
        elitism_size: int,
        tournument_size: int,
        rng: random.Random
) -> list[Route]:
    selection_results = []
    for i in range(elitism_size):
        selection_results.append(population[ranked_population[i][0]])

    if selection_method == "roulette":
        fitness_sum = sum(fitness for _, fitness in ranked_population)
        probabilities = [fitness / fitness_sum for _, fitness in ranked_population]
        selected_indices = rng.choices(
            [idx for idx, _ in ranked_population],
            weights=probabilities,
            k=len(population) - elitism_size
        )
        for idx in selected_indices:
            selection_results.append(population[idx])
        return selection_results
    elif selection_method == "tournament":
        for _ in range(len(population) - elitism_size):
            tournament = rng.sample(ranked_population, tournument_size)
            winner = max(tournament, key=lambda x: x[1])
            selection_results.append(population[winner[0]])
    return selection_results

# Order Crossover (OX)
def crossover(parent1: Route, parent2: Route, rng: random.Random) -> Route:
    child: list[int | None] = [None] * len(parent1)
    start, end = sorted(rng.sample(range(len(parent1)), 2))

    child[start:end + 1] = parent1[start:end + 1]

    parent2_genes = [item for item in parent2 if item not in child]

    idx = 0
    for i in range(len(child)):
        if child[i] is None:
            child[i] = parent2_genes[idx]
            idx += 1

    return child


def breed_population(mating_pool: list[Route], elitism_size: int, rng: random.Random) -> list[Route]:
    children = []

    children.extend(mating_pool[:elitism_size])

    pool = rng.sample(mating_pool, len(mating_pool))
    for i in range(len(mating_pool) - elitism_size):
        child = crossover(pool[i], pool[len(mating_pool) - i - 1], rng)
        children.append(child)

    return children

def mutate(route: Route, mutation_rate: int, rng: random.Random) -> Route:
    if rng.random() < mutation_rate:
        i, k = sorted(rng.sample(range(len(route)), 2))
        return two_opt_swap(route, i, k)
    return route


def mutate_population(population: list[Route], mutation_rate: int, rng: random.Random) -> list[Route]:
    mutated_pop = [population[0]]
    for route in population[1:]:
        mutated_pop.append(mutate(route, mutation_rate, rng))
    return mutated_pop


def genetic_algorithm(configuration: dict, selection_method: str, dist_matrix: np.ndarray, rng: random.Random):
    population_size = configuration["POPULATION_SIZE"]
    mutation_rate = configuration["MUTATION_RATE"]
    tournament_size = configuration["TOURNAMENT_SIZE"]
    generations = configuration["GENERATIONS"]
    elitism_size = math.ceil(population_size / 10)
    population = initial_population(population_size, dist_matrix, rng)

    best_route = None
    best_distance = float('inf')

    print("Iniciando evolução...")

    for gen in range(generations):
        ranked_pop = rank_routes(population, dist_matrix)
        current_best_idx, _ = ranked_pop[0]
        current_best_dist = route_length(population[current_best_idx], dist_matrix)

        if current_best_dist < best_distance:
            best_distance = current_best_dist
            best_route = population[current_best_idx]

        selected = selection(selection_method, ranked_pop, population, elitism_size, tournament_size, rng)
        children = breed_population(selected, elitism_size, rng)
        population = mutate_population(children, mutation_rate, rng)

    return best_route, best_distance


def main():
    rng = random.Random()

    pts = load_points_from_csv(FILE)
    dist_matrix = distance_matrix(pts)
    all_run_results = []
    selection_methods = ["tournament", "roulette"]
    for selection_method in selection_methods:
        for configuration in configurations:
            print(
                f"Executando: Selection method='{selection_method}', Config='{configuration['name']}'..."
            )
            for i in range(50):
                start_time = time.time()
                best_route, best_distance = genetic_algorithm(configuration, selection_method, dist_matrix, rng)
                end_time = time.time()
                execution_time = end_time - start_time

                all_run_results.append(
                    {
                        "route": best_route,
                        "length": best_distance,
                        "time": execution_time,
                        "selection_method": selection_method,
                        "config_name": configuration["name"],
                    }
                )
                print(
                    f"  -> Execução {i+1:02d}: Comprimento = {best_distance:.4f}, Tempo = {execution_time:.4f}s"
                )
    if not all_run_results:
        print("Nenhum resultado para analisar.")
        return

    best_overall_result, worst_overall_result = get_best_and_worst_results(all_run_results)

    print_overall_results(best_overall_result, worst_overall_result, "selection_method")

    export_results_to_csv(all_run_results, "genetic_algorithm")

    if PLOT:
        plot_results(
            pts,
            best_overall_result,
            worst_overall_result,
            all_run_results,
            selection_methods,
            "selection_method",
            color_map
        )
        plt.show()


if __name__ == '__main__':
    main()
