# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Pick and place in basket manipulation environment.

This environment involves picking up a cube from a table and placing it
into a basket. The cube and basket positions are randomized at reset.

Registered environments:
    - Isaac-Pick-Place-Basket-Franka-v0: Joint position control
    - Isaac-Pick-Place-Basket-Franka-IK-Rel-v0: IK relative pose control
"""

from .pick_place_basket_env_cfg import PickPlaceBasketEnvCfg

# Import config to register gym environments
from .config import *  # noqa: F401, F403

