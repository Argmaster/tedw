from dataclasses import dataclass
from optmath import RecordBase, autoscale
from optmath.Kmeans import Kmeans
from optmath.PCA import PCA
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


@dataclass(frozen=True)
class PostPCASeed(RecordBase):
    pc1: float
    pc2: float
    pc3: float


class Session:
    def __init__(self, source: str) -> None:
        raw = pandas.read_csv(source).to_numpy()
        self.data = PumpkinSeed.new(autoscale(raw))

    def kmeans(self):
        return Kmeans(self.data, 3)

    def post_pca_kmeans(self):
        subview = PCA(self.data).get_view_from_first_top(3)
        data = PostPCASeed.new(subview.get_post_transfrom())
        view = Kmeans(data, cluster_count=2)
        return view
