import numpy as np


def transpose(M: np.array) -> np.array:
    return np.array(
        [[M[i, j] for i in range(M.shape[0])] for j in range(M.shape[1])]
    )


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
    # Nie, macierze są tylko dwuwymiarowe, ndarray może być n-wymiarowa
    print(m4x4)
    # 6
    print(np.array([m4x4[i, i] for i in range(4)]))
    # 7

    print("m4x4:", m4x4, sep="\n")
    print("m4x4_T:", transpose(m4x4), sep="\n")


if __name__ == "__main__":
    task_1()
