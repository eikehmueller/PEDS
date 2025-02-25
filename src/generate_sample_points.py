import numpy as np
from matplotlib import pyplot as plt

n = 10
delta = 0.2
delta_boundary = 0.05

rng = np.random.default_rng(seed=2351254)

points = []

while len(points) < n:
    p_new = rng.uniform(low=0, high=1, size=(2))
    # discard points that are too close to the boundary of the domain
    if (
        (p_new[0] < delta_boundary)
        or (1 - p_new[0] < delta_boundary)
        or (p_new[1] < delta_boundary)
        or (1 - p_new[1] < delta_boundary)
    ):
        continue
    for p in points:
        if (p_new[0] - p[0]) ** 2 + (p_new[1] - p[1]) ** 2 < delta**2:
            break
    else:
        points.append(p_new)

points = np.asarray(points)

plt.clf()
plt.plot(
    points[:, 0], points[:, 1], linewidth=0, marker="o", markersize=4, color="blue"
)
ax = plt.gca()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.savefig("points.pdf", bbox_inches="tight")

print("[", end="")
for j, p in enumerate(points):
    print(f"[{p[0]:.3f},{p[1]:.3f}]", end="," if j < len(points) - 1 else "")
print("]")
