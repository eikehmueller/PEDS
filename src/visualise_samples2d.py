import argparse
import itertools
import numpy as np
from peds.auxilliary import save_vtk
from peds.distributions_lognormal import LogNormalDistribution2d
from peds.distributions_fibres import FibreDistribution2d, FibreRadiusDistribution

parser = argparse.ArgumentParser()
parser.add_argument(
    "--distribution",
    type=str,
    action="store",
    help="ditribution to draw from",
    choices=["fibres", "lognormal"],
    default="fibres",
)

parser.add_argument("--n", type=int, action="store", help="grid size", default=128)

parser.add_argument(
    "--nsamples", type=int, action="store", help="number of samples", default=8
)

parser.add_argument(
    "--Lambda", type=float, action="store", help="correlation length", default=0.1
)


args, _ = parser.parse_known_args()

if args.distribution == "lognormal":
    distribution = LogNormalDistribution2d(args.n, args.Lambda)
else:
    distribution = FibreDistribution2d(
        args.n,
        volume_fraction=0.55,
        r_fibre_dist=FibreRadiusDistribution(
            r_avg=7.5e-3, r_min=5.0e-3, r_max=10.0e-3, sigma=0.5e-3, gaussian=True
        ),
    )

samples = [sample for sample in list(itertools.islice(distribution, args.nsamples))]
save_vtk(samples, "samples.vtk")
