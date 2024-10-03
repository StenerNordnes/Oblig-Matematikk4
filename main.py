import random
from functools import lru_cache
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

L: int = 20
T: int = 400

alpha: int = 4
dx: int = 1

dt: float = (dx**2) / (4 * alpha)
gamma: float = (alpha * dt) / (dx**2)

u: np.ndarray = np.zeros((T, L, L))  # løsning


# forskjellige initialbetingelser
def sin_initial(i, j):
    return np.sin(np.pi * i * dx / L) * np.sin(np.pi * j * dx / L)


def cos_initial(i, j):
    return np.cos(np.pi * i * dx / L) * np.cos(np.pi * j * dx / L)


def rampe_initial(i, j):
    return i * dx / L


def dirac_initial(i, j):
    return 2 if i == L // 2 and j == L // 2 else 0


def random_initial(i, j):
    return random.random()


def initial(func: Callable[[int, int], float]) -> None:
    for i in range(L):
        for j in range(L):
            u[0, i, j] = func(i, j)


# fylle initialbetingelser
initial(rampe_initial)


@lru_cache(maxsize=None)
def varme_eksplisitt() -> None:
    for k in range(0, T - 1):
        for i in range(1, L - 1):
            for j in range(1, L - 1):
                u[k + 1, i, j] = (
                    gamma
                    * (
                        u[k, i + 1, j]
                        + u[k, i - 1, j]
                        + u[k, i, j + 1]
                        + u[k, i, j - 1]
                        - 4 * u[k, i, j]
                    )
                    + u[k, i, j]
                )


varme_eksplisitt()

X, Y = np.meshgrid(np.arange(0, L), np.arange(0, L))

fig: plt.Figure = plt.figure()
ax: plt.Axes = fig.add_subplot(111, projection="3d")


def update(frame) -> plt.Axes:
    ax.clear()
    ax.set_zlim(0, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.plot_surface(X, Y, u[frame, :, :])
    return ax


anim: FuncAnimation = FuncAnimation(fig, update, frames=T, interval=10)

anim.save("animation.gif", writer="imagemagick", fps=30)

plt.show()
