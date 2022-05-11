import math
from pathlib import Path
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


FILE_DIR = Path(__file__).parent

A_TABLE = [
    0.4254,
    0.2944,
    0.2487,
    0.2148,
    0.187,
    0.163,
    0.1415,
    0.1219,
    0.1036,
    0.0862,
    0.0697,
    0.0537,
    0.0381,
    0.0227,
    0.0076,
]


class Session:

    raw_data: pd.DataFrame

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        self.raw_data = pd.read_csv(FILE_DIR / "raw_data.csv")

    @staticmethod
    def bucket_plot(
        data: List[float], explicit_bucket_count: int = None
    ) -> Tuple[List[str], List[float]]:
        """Classifies given data into len(data) // 4 + 1 categories,
        returns list of category range labels and list of list of values in
        categories.
        """
        max_val = max(data) * 1.05
        min_val = min(data) * 0.95
        if explicit_bucket_count is None:
            bucket_count: int = len(data) // 4 - 2
        else:
            bucket_count = explicit_bucket_count
        span: float = abs(max_val - min_val)
        bucket_size: float = span / bucket_count
        buckets: Dict[Tuple[float, float], List[float]] = {}

        for i in range(bucket_count):
            bucket_min = min_val + i * bucket_size
            bucket_max = min_val + (i + 1) * bucket_size
            buckets[bucket_min, bucket_max] = []
        else:
            last_bucket = buckets[bucket_min, bucket_max]

        for value in data:
            # small faux-pass here: order is not guaranteed on every runtime
            # environment however with CPython 3.6+ dictionary impl keeps
            # order of insertions.
            # MAY be a problem with Jython or PyPy or older CPython versions
            # Btw see https://www.hyrumslaw.com/
            for (bucket_min, bucket_max), bucket in buckets.items():
                if bucket_min <= value <= bucket_max:
                    bucket.append(value)
                    break
            else:
                last_bucket.append(value)

        return list(buckets.keys()), list(buckets.values())

    @staticmethod
    def draw_in_buckets_manual(column, bucket_count=None):
        labels, values = Session.bucket_plot(column, bucket_count)
        bar_width = (labels[0][1] - labels[0][0]) * 0.7
        center_labels = [sum(label) / 2 for label in labels]

        plt.figure(figsize=(10, 6), dpi=100)
        plt.bar(
            center_labels,
            height=[len(v) for v in values],
            width=bar_width,
            color="gray",
        )
        plt.axvline(np.mean(column), c="r")
        plt.axvline(np.median(column), c="y")
        plt.axvline(np.quantile(column, 0.25), c="g")
        plt.axvline(np.quantile(column, 0.75), c="g")
        plt.xticks(
            center_labels,
            [f"<{round(lbl[0], 3)}; {round(lbl[1], 3)})" for lbl in labels],
        )

    @staticmethod
    def draw_in_buckets(data):
        n, bins = np.histogram(data, bins=len(data) // 4)

        bins_rounded = []
        for value in bins:
            bins_rounded.append(round(value, 2))

        sns.histplot(data, bins=bins_rounded, kde=True)
        plt.axvline(np.mean(data), c="r")
        plt.axvline(np.median(data), c="y")
        plt.axvline(np.quantile(data, 0.25), c="g")
        plt.axvline(np.quantile(data, 0.75), c="g")
        plt.xticks(bins_rounded)

    @staticmethod
    def is_normal_dist(data_column):
        column = data_column.sort_values()
        mean = column.mean()
        data_list = column.tolist()
        S_sq = sum((x - mean) ** 2 for x in data_list)
        m = math.floor(len(data_list) / 2)
        b = sum(
            A_TABLE[i] * (data_list[-1 - i] - data_list[i]) for i in range(m)
        )
        W = b**2 / S_sq
        return W > 0.927

    def nth_column_props(self, no: int):
        column_tag = self.raw_data.columns[no]
        column = self.raw_data[column_tag]
        print(column.describe())
        if self.is_normal_dist(column):
            print("Dane mają rozkład normalny")
        else:
            print("Dane nie mają rozkładu normalnego")
        self.draw_in_buckets_manual(column)
        plt.title(f"{column_tag} - przed transformacją")
        plt.show()

    def show_post_transform_table(self):
        self.post_proc_data = self.raw_data.copy(True)

        column = self.post_proc_data["Solidity"].sort_values()
        plt.figure(figsize=(10, 6), dpi=100)
        Session.draw_in_buckets(column)
        plt.title("Solidity - przed transformacją")
        plt.show()

        column = np.array([np.log10(x * 0.5) for x in column])  # 2
        plt.figure(figsize=(10, 6), dpi=100)
        Session.draw_in_buckets(column)
        plt.title("Solidity - po transformacji")
        plt.show()

        self.post_proc_data["Solidity"] = column
        self.post_proc_data = self.post_proc_data.rename(
            {
                "Solidity": "Solidity*",
                "Area": "Area",
                "Perimeter": "Perimeter",
                "Major_Axis_Length": "Major_Axis_Length",
                "Minor_Axis_Length": "Minor_Axis_Length",
                "Roundness": "Roundness",
            },
            axis=1,
        )
        return self

    def save_post_transform_to_file(self):
        self.post_proc_data.to_csv(FILE_DIR / "post_transform_data.csv")

    def heatmap(self):
        def cor(x: np.array, y: np.array) -> float:
            x_mean = x.mean()
            y_mean = y.mean()
            return sum(
                (x_i - x_mean) * (y_i - y_mean) for x_i, y_i in zip(x, y)
            ) / np.sqrt(
                sum((x_i - x_mean) ** 2 for x_i in x)
                * sum((y_i - y_mean) ** 2 for y_i in y)
            )

        N = len(self.post_proc_data.columns)

        cor_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                cor_matrix[i, j] = cor(
                    self.post_proc_data.iloc[:, i],
                    self.post_proc_data.iloc[:, j],
                )
        for c in cor_matrix:
            for r in c:
                print(f"{r:> 4.3f} ", end="")
            print()
        plt.figure(figsize=(8, 6), dpi=100)
        sns.heatmap(
            cor_matrix,
            annot=True,
            xticklabels=self.post_proc_data.columns,
            yticklabels=self.post_proc_data.columns,
        )

    def correlation_plots(self):
        N = len(self.post_proc_data.columns)
        labels = self.post_proc_data.columns
        for i in range(N):
            for j in range(N):
                if i > j:
                    plt.figure(figsize=(6, 6), dpi=100)
                    plt.title(f"{labels[i]} - {labels[j]}")
                    plt.xlabel(labels[i])
                    plt.ylabel(labels[j])
                    plt.scatter(
                        self.post_proc_data.iloc[:, i],
                        self.post_proc_data.iloc[:, j],
                    )
                else:
                    break
