from dataclasses import dataclass
from functools import cached_property
from matplotlib import pyplot as plt
from optmath.HCA import RecordBase, Cluster, HCA, CompleteLinkage, Euclidean
from optmath.HCA.record import autoscale
from scipy.cluster import hierarchy

import pandas


@dataclass(frozen=True)
class PumpkinSeed(RecordBase):
    Area: float
    Perimeter: float
    Major_Axis_Length: float
    Minor_Axis_Length: float
    Solidity: float
    Roundness: float


@dataclass
class Session:

    csv_data_path: str

    @cached_property
    def shared_data(self) -> pandas.DataFrame:
        data_frame = pandas.read_csv(self.csv_data_path)
        return data_frame

    @property
    def unique_data(self) -> pandas.DataFrame:
        data_frame = pandas.read_csv(self.csv_data_path)
        return data_frame

    @cached_property
    def autoscaled_data(self) -> pandas.DataFrame:
        return autoscale(self.shared_data.to_numpy())

    def custom_HCA(self):
        clusters = Cluster.new(PumpkinSeed.new(self.autoscaled_data))
        algorithm = HCA(clusters, CompleteLinkage(Euclidean()))
        cluster = algorithm.result()
        return cluster.Z()

    def scipy_HCA(
        self,
        method: str = "complete",
        metric: str = "euclidean",
    ):
        return hierarchy.linkage(
            self.autoscaled_data,
            method=method,
            metric=metric,
        )

    def dendrogram(self, z):
        plt.figure(figsize=(9, 7), dpi=100)
        hierarchy.dendrogram(z, leaf_rotation=90.0, leaf_font_size=8.0)
        plt.title("HCA outcome dendrogram")
        plt.ylabel("Distance")
        plt.xlabel("Cluster identifier")
        plt.show()

    @cached_property
    def distance_matrix(self):
        clusters = Cluster.new(PumpkinSeed.new(self.autoscaled_data))
        algorithm = HCA(clusters, CompleteLinkage(Euclidean()))
        return pandas.DataFrame(algorithm.step.distance_matrix)

    def dendrogram_grid(self):
        fig, axes = plt.subplots(3, 3)
        fig: plt.Figure
        fig.set_size_inches(32, 32)
        fig.set_dpi(200)
        i = 0
        for row, method in zip(axes, ("single", "complete", "ward")):
            for ax, metric in zip(
                row, ("euclidean", "cityblock", "chebyshev")
            ):
                ax: plt.Axes
                try:
                    z = self.scipy_HCA(method, metric)
                    hierarchy.dendrogram(
                        z, leaf_rotation=90.0, leaf_font_size=8.0, ax=ax
                    )
                    ax.set_title(f"{i}: HCA method={method} metric={metric}")
                    ax.set_ylabel("Distance")
                    ax.set_xlabel("Cluster identifier")
                except Exception:
                    ax.text(
                        0.3,
                        0.45,
                        "In SciPy Ward works only\nwith Euclidean distance!",
                        color="red",
                        fontsize="xx-large",
                    )
                i += 1
