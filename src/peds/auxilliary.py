__all__ = ["save_vtk"]


def save_vtk(alpha_samples, domain_size, filename):
    """Save a list of samples to vtk file

    :arg alpha_samples: list of fields
    :arg filename: name of file to write to
    """
    alpha = alpha_samples[0]
    nx = alpha.shape[0] - 1
    ny = alpha.shape[1] - 1
    npoints = (nx + 1) * (ny + 1)
    hx = domain_size / nx
    hy = domain_size / ny
    with open(filename, "w", encoding="utf8") as f:
        print("# vtk DataFile Version 2.0", file=f)
        print("Sample state", file=f)
        print("ASCII", file=f)
        print("DATASET STRUCTURED_POINTS", file=f)
        print(f"DIMENSIONS {nx + 1} {ny + 1} 1 ", file=f)
        print(f"ORIGIN -{0.5*domain_size} -{0.5*domain_size} 0.0", file=f)
        print(f"SPACING {hx} {hy} 0", file=f)
        print(file=f)
        print(f"POINT_DATA {npoints}", file=f)
        for ell, alpha in enumerate(alpha_samples):
            print(f"SCALARS sample_{ell:03d} double 1", file=f)
            print("LOOKUP_TABLE default", file=f)
            for j in range(nx + 1):
                for k in range(nx + 1):
                    print(f"{alpha[j,k]:12.8e}", file=f)
