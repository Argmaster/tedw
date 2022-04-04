from dataclasses import dataclass
from math import sqrt
from typing import Callable, List


def filter_non_numeric(data: List[List[float]]):
    return [
        [cell for cell in row if isinstance(cell, (float, int))]
        for row in data
    ]


def autonormalize(data: List[List[float]]):
    rows = len(data)
    cols = len(data[0])
    normaizers = []
    for j in range(cols):
        mean = sum(data[i][j] for i in range(rows)) / rows
        std = sqrt(sum((data[i][j] - mean) ** 2 for i in range(rows)) / rows)
        normaizers.append((mean, std))

    return [
        [(cell - mean) / std for cell, (mean, std) in zip(row, normaizers)]
        for row in data
    ]


def euclidean(row_0: List[float], row_1):
    return sqrt(
        sum((cell_0 - cell_1) ** 2 for cell_0, cell_1 in zip(row_0, row_1))
    )


@dataclass
class DistanceMatrix:
    data: List[List[float]]
    distance: Callable
    distance_selector: Callable
    matrix: List[List[float]] = None

    def __post_init__(self):
        if self.matrix is None:
            self.matrix = []
            for i, row_0 in enumerate(self.data):
                self.matrix.append(
                    [
                        self.distance(row_0, row_1)
                        for j, row_1 in enumerate(self.data)
                        if j < i
                    ]
                )

    def reduce(self) -> "DistanceMatrix":
        self.group = self._select_group()
        matrix_new = [
            [c for j, c in enumerate(r) if j not in self.group]
            for i, r in enumerate(self.matrix)
            if i not in self.group
        ]
        matrix_new.append(
            [
                self.distance_selector(
                    [cell for j, cell in enumerate(row) if j in self.group]
                )
            ]
            for i, row in enumerate(self.matrix)
            if i not in self.group
        )
        return DistanceMatrix(
            None, self.distance, self.distance_selector, matrix_new
        )

    def _select_group(self) -> List[int]:
        min_distance = min(min(row) for row in self.matrix[1:])
        group = set()
        for i, row in enumerate(self.matrix):
            for j, cell in enumerate(row):
                if i == j:
                    continue
                if cell == min_distance:
                    group.add(i)
                    group.add(j)
        return group

    def __str__(self):
        return "\n".join(
            ", ".join(f"{cell: >5.3f}" for cell in row) for row in self.matrix
        )


def HCA(data: List[List[float]], distance=euclidean, distance_selector=max):

    dm = DistanceMatrix(data, distance, distance_selector)
    dm1 = dm.reduce()
    print(dm.group)
    print(dm1)
