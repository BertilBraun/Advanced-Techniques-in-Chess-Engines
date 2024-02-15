import numpy as np
from numpy.typing import NDArray


def lerp(a, b, t: float) -> NDArray[np.float32]:
    return t * a + (1 - t) * b
