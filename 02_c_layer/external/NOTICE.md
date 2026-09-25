# Third-party code notice (02_c_layer/external/)

This directory vendors baseline implementations from the SHREC 2025 protein
shape classification track for reproducibility. All copyrights remain with the
original authors.

| Directory | Origin | License |
|-----------|--------|---------|
| `shrec2025_protein/` | Official SHREC 2025 track baseline repository (Atsushi Tatsuma) | MIT (see `LICENSE.txt`) |
| `SHREC2025/` | Track organizers' 3D-Zernike baseline (`map2zernike` / `obj2grid` binaries, `vtk2zd.py`), distributed to participants via the track site | No explicit license file; reproduced here for reproducibility of the published comparison only |
| `guerra/` | Rotation-invariant histogram descriptor code accompanying the track report (M. Guerra et al.) | No explicit license file; same status as above |
| `tehrani/` | Track participant baseline (image-based) | No explicit license file; same status as above |

If you redistribute this package, keep this notice. Contact the original
authors for any use beyond reproducing the results of the accompanying paper.
