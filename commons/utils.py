import math
import random

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


# 2-Opt operações


def two_opt_swap(route: Route, i: int, k: int) -> Route:
    # Reverte o segmento [i:k]
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


def random_points(n: int, rng: random.Random) -> list[Point]:
    return [(rng.random(), rng.random(), str(i)) for i in range(n)]


def plot_route(pts, route, title=None, show_ids=False):
    # Extract coordinates
    xs = [pts[i][0] for i in route]
    ys = [pts[i][1] for i in route]

    plt.figure(figsize=(8, 6))
    plt.plot(xs + [xs[0]], ys + [ys[0]], 'o-', markersize=8)  # linha do percurso com pontos
    plt.plot(xs[0], ys[0], 'o', markersize=10, color='green', label='Início')

    # Adiciona o nome das cidades (id) ao lado de cada ponto
    for idx in route:
        x, y = pts[idx][0], pts[idx][1]
        cidade = pts[idx][2]
        plt.text(x, y, str(cidade), fontsize=9, color='red', ha='right', va='bottom')

    if title:
        plt.title(title)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(min(xs) - 0.5, max(xs) + 0.5)
    plt.ylim(min(ys) - 0.5, max(ys) + 0.5)
