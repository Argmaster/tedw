import numpy as np


def task_1():
    # 1
    A = np.linspace(1, 20, 10)
    # 2
    B = np.linspace(0, 1, 10)
    # 3
    AB = A * B
    print(AB)
    # 4
    c = 2
    Ac = A * c
    print(Ac)
    # 5
    m4x4 = np.matrix(np.random.randint(1, 101, (4, 4)))
    # Nie, macierze są tylko dwuwymiarowe, ndarray jest n-wymiarowa
    print(m4x4)
    # 6
    cross = np.array([m4x4[i, i] for i in range(4)])
    print(cross)
    # 7
    transposed = [[e] for e in cross]
    print(transposed)


if __name__ == "__main__":
    task_1()
