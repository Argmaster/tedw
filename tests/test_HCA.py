from lab3 import HCA, autonormalize, filter_non_numeric
from lab3.alt import Other, PumpkinSeed


def test_HCA_euk_max():
    data = [
        ["A", 0, 2],
        ["B", 3, 4],
        ["C", 5, 6],
    ]
    data = filter_non_numeric(data)
    data = autonormalize(data)
    for row in data:
        print(row)
    HCA(data)


def test_Other():
    ps = PumpkinSeed(a=1, b=10)
    o = Other(1, 1.0, 2)
    print(ps, o)
