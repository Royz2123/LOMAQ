import math
import numpy as np

CARTPOLES = 3
CARTDIST = 2

X_MARGIN = 6
THETA_THRESHOLD_RADIANS = 12 * 2 * math.pi / 360

HIGH = np.array([X_MARGIN * 2,
                 np.finfo(np.float32).max,
                 THETA_THRESHOLD_RADIANS * 2,
                 np.finfo(np.float32).max],
                dtype=np.float32)

# how many cartpoles does one see right and left? 0 means sees just himself
OBSERVATION_RADIUS = 0

GRAVITY = 9.8
