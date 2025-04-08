import argparse
import itertools
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
            r_avg=0.0375, r_min=0.025, r_max=0.05, sigma=0.0025, gaussian=True
        ),
        fast_code=True,
    )

samples = [sample for sample in list(itertools.islice(distribution, args.nsamples))]
save_vtk(samples, "samples.vtk")
