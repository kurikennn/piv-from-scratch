import numpy as np
import pytest

from pivfs.flows import LambOseen
from pivfs.metrics import interior_mask, rms, vector_errors
from pivfs.piv import VectorField

SHAPE = (512, 512)


def interior_errors(vf: VectorField, flow) -> np.ndarray:
    """Error magnitudes at interior vectors."""
    return vector_errors(vf, flow).magnitude[interior_mask(vf, SHAPE)]


def core_rms(vf: VectorField, flow: LambOseen) -> float:
    """RMS error magnitude of interior vectors inside r < 2 r_c."""
    r = np.hypot(vf.x - flow.x0, vf.y - flow.y0)
    core = interior_mask(vf, SHAPE) & (r < 2 * flow.r_c)
    return rms(vector_errors(vf, flow).magnitude[core])


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
