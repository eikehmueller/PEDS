import numpy as np
from matplotlib import pyplot as plt

n = 10
domain_size = 0.2
delta = 0.04
delta_boundary = 0.01

rng = np.random.default_rng(seed=2351254)

points = []

while len(points) < n:
    p_new = rng.uniform(low=0, high=domain_size, size=(2))
    # discard points that are too close to the boundary of the domain
    if (
        (p_new[0] < delta_boundary)
        or (domain_size - p_new[0] < delta_boundary)
        or (p_new[1] < delta_boundary)
        or (domain_size - p_new[1] < delta_boundary)
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
ax.set_xlim(0, domain_size)
ax.set_ylim(0, domain_size)
plt.savefig("points.pdf", bbox_inches="tight")

print("[", end="")
for j, p in enumerate(points):
    print(f"[{p[0]:.3f},{p[1]:.3f}]", end="," if j < len(points) - 1 else "")
print("]")
