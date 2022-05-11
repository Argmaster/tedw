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
