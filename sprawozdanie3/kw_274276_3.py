from dataclasses import dataclass
from optmath import PCAResutsView, RecordBase, PCA, autoscale
from numpy.typing import NDArray
import numpy as np


import pandas


@dataclass(frozen=True)
class PumpkinSeed(RecordBase):
    Area: float
    Perimeter: float
    Major_Axis_Length: float
    Minor_Axis_Length: float
    Convex_Area: int
    Equiv_Diameter: float
    Eccentricity: float
    Solidity: float
    Extent: float
    Roundness: float
    Aspect_Ration: float
    Compactness: float
    Class: str


@dataclass
class Session:

    csv_data_path: str

    @property
    def unique_data(self) -> pandas.DataFrame:
        data_frame = pandas.read_csv(self.csv_data_path)
        return data_frame

    @property
    def autoscaled_data(self) -> NDArray[np.float64]:
        return autoscale(self.unique_data.to_numpy())

    def run_PCA(self) -> PCAResutsView:
        return PCA(PumpkinSeed.new(self.autoscaled_data))

    def view_c(self) -> None:
        org = PCA(PumpkinSeed.new(self.autoscaled_data))

        view = org.get_view_from_kaiser_criteria()
        view.nd_data = view.nd_data[:1302]
        fig, ax = view.show_principal_component_grid(20, color="#1f49bf01")

        view2 = org.get_view_from_kaiser_criteria()
        view2.nd_data = view2.nd_data[1302:]
        fig, ax = view2.show_principal_component_grid(
            10, color="#ad1c4501", fig=fig, axes=ax
        )

        fig.set_size_inches(20, 14)
