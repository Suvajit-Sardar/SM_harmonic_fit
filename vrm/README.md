# VRM / VRMA (vendored, third-party)

`VRM_VRMA.py` in this directory is vendored **unmodified** from:

https://github.com/MatteoStraccamore/VRM_VRMA

which implements the Velocity Ring Model (VRM) and Velocity Ring Model with
Arcs (VRMA) described in:

> Sylos Labini, F., Straccamore, M., De Marzo, G., & Comerón, S.,
> "Mapping non-axisymmetric velocity fields of external galaxies"
> ApJ 988, 122 (2025), doi:10.3847/1538-4357/adc71c
> (arXiv/MNRAS preprint: arXiv:2306.12902)

**License note:** the upstream repository has no LICENSE file at the time of
vendoring (2026-08). The code is reproduced here for a research comparison
with clear attribution to the authors and source repository; no license
grant beyond that is implied.

## What VRM/VRMA is, and how it differs from this project's own pipeline

VRM fits an ordinary-least-squares model of the same functional form as this
project's harmonic fit,

```
v_los(R, theta) = [v_t(R) cos(theta) + v_r(R) sin(theta)] sin(i) + v_sys
```

but where `harmonic_fit.py` **fixes** the rotation term at Barolo's `VROT`
and only fits `c0` (offset) and `s1` (= `v_r`), VRM fits `v_t` **and** `v_r`
together, per ring, with nothing held fixed except the global geometry
(inclination `i` and position angle). `v_sys` is itself fit once, globally,
by treating the whole map as a single ring (`VRM._OFF`), rather than taken
from the ringlog.

VRMA extends VRM by splitting each ring into azimuthal arcs, fitting an
independent `(v_t, v_r)` per `(ring, arc)` cell instead of per ring — this
recovers a full 2D map of both velocity components rather than a 1D radial
profile, at the cost of fewer points (and therefore more noise) per cell.

Ring geometry is also handled differently: VRM bins in **rescaled** radius
`(R - R_min) / (R_max - R_min)` over the full extent of whatever points are
passed in, not fixed physical ring edges from a tilted-ring model. There is
no equivalent of this project's bootstrap, PA/VSYS degeneracy scan, or
leakage diagnostics — `VRM._matrix()` returns point estimates only, with no
uncertainty quantification of any kind.

**Known upstream quirk:** the constructor accepts a `phi0input` (position
angle) argument, but it is stored and never used anywhere in
`_rescaled_R_theta`. The method silently assumes the input `(x, y)` are
already rotated into a frame where the kinematic major axis lies along
`+x` (the paper confirms this is intentional — the recommended workflow is
to pre-rotate the data by the position angle before calling VRM, not to
rely on `phi0input`). `vrm/bridge.py` in this project does that rotation
explicitly, using the same PA/inclination convention as `harmonic_fit.py`,
so this is handled correctly here — but it is easy to get silently wrong if
you use the upstream class directly and expect `phi0input` to do something.

See `../vrm_pipeline.ipynb` for the actual comparison against this
project's own `s1(R)` result.
