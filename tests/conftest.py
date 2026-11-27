import numpy as np
import pytest
import warp as wp

wp.config.quiet = True
wp.init()


@pytest.fixture
def device():
    return "cpu"


def log_params(rigidity, device="cpu", grad=False):
    return wp.array(np.log(rigidity).astype(np.float32), dtype=float, device=device, requires_grad=grad)
