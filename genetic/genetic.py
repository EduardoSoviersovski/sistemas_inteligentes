import math
import random
import numpy as np
import matplotlib.pyplot as plt

from commons.utils import (
    route_length,
    load_points_from_csv,
    distance_matrix,
    plot_route,
    two_opt_swap,
    Route, create_random_route
)
POPULATION_SIZE = 30
ELITISM_SIZE = math.ceil(POPULATION_SIZE / 10)
MUTATION_RATE = 0.015
TOURNAMENT_SIZE = 5
GENERATIONS = 500
FILE = "files/50.csv"
PLOT = True

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

#TODO: Testar com roleta russa. Testar removendo um elemento da lista caso dele seja selecionado (evitar duplicados)
def selection(
        ranked_population: list[tuple[int, float]],
        population: list[Route],
        rng: random.Random
) -> list[Route]:
    selection_results = []
    for i in range(ELITISM_SIZE):
        selection_results.append(population[ranked_population[i][0]])

    for _ in range(len(population) - ELITISM_SIZE):
        tournament = rng.sample(ranked_population, TOURNAMENT_SIZE)
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


def breed_population(mating_pool: list[Route], rng: random.Random) -> list[Route]:
    children = []

    children.extend(mating_pool[:ELITISM_SIZE])

    pool = rng.sample(mating_pool, len(mating_pool))
    for i in range(len(mating_pool) - ELITISM_SIZE):
        child = crossover(pool[i], pool[len(mating_pool) - i - 1], rng)
        children.append(child)

    return children

# TODO: Testar aumentar o mutation rate a cada geracao
def mutate(route: Route, rng: random.Random) -> Route:
    if rng.random() < MUTATION_RATE:
        i, k = sorted(rng.sample(range(len(route)), 2))
        return two_opt_swap(route, i, k)
    return route


def mutate_population(population: list[Route], rng: random.Random) -> list[Route]:
    mutated_pop = [population[0]]
    for route in population[1:]:
        mutated_pop.append(mutate(route, rng))
    return mutated_pop


def genetic_algorithm(dist_matrix: np.ndarray, rng: random.Random):
    population = initial_population(POPULATION_SIZE, dist_matrix, rng)

    best_route = None
    best_distance = float('inf')

    print("Iniciando evolução...")

    for gen in range(GENERATIONS):
        ranked_pop = rank_routes(population, dist_matrix)
        current_best_idx, _ = ranked_pop[0]
        current_best_dist = route_length(population[current_best_idx], dist_matrix)

        if current_best_dist < best_distance:
            best_distance = current_best_dist
            best_route = population[current_best_idx]

        if (gen + 1) % 10 == 0:
            print(f"Geração {gen + 1}: Melhor Distância = {best_distance:.4f}")

        selected = selection(ranked_pop, population, rng)
        children = breed_population(selected, rng)
        population = mutate_population(children, rng)

    return best_route, best_distance


def main():
    rng = random.Random()

    pts = load_points_from_csv(FILE)
    dist_matrix = distance_matrix(pts)
    best_route, best_distance = genetic_algorithm(dist_matrix, rng)

    print("\nEvolução finalizada!")
    print(f"Melhor rota encontrada: {best_route}")
    print(f"Distância: {best_distance:.4f}")

    if PLOT:
        plot_route(pts, best_route, f"AG – melhor rota {best_distance:.4f}", show_ids=True)
        plt.show()


if __name__ == '__main__':
    main()
