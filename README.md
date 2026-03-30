# (Tensor) MP-PCA

MP-PCA denoiser for MRI data compatible with the tensor method [2] — Python package with a fast C++/Eigen backend.

**References:** 
1. Veraart J, Novikov D, Christiaens D, Ades-aron B, Sijbers J, Fieremans E, *Denoising of diffusion MRI using random matrix theory*, NeuroImage 2016; 394-406. [doi:10.1016/j.neuroimage.2016.08.016](https://doi.org/10.1016/j.neuroimage.2016.08.016)
2. Olesen JL, Ianus, A, Østergaard, L, Shemesh, N, Jespersen SN, *Tensor denoising of multidimensional MRI data*, Magn Reson Med. 2022; 1-13. [doi:10.1002/mrm.29478](https://doi.org/10.1002/mrm.29478)
3. Fernandes F, Olesen JL, Jespersen SN, Shemesh N, *MP-PCA denoising of fMRI time-series data can lead to artificial activation “spreading”*, NeuroImage 2023; 120118 [doi:10.1016/j.neuroimage.2023.120118](https://doi.org/10.1016/j.neuroimage.2023.120118)

---

## Installation

```bash
git clone https://github.com/lsoustelle/tmppca.git
cd tmppca
conda create -n tmppca -c conda-forge python=3.11 numpy scipy nibabel pybind11 eigen cmake gxx_linux-64 openmp scikit-build-core ninja -y
conda activate tmppca
pip install .
```

---

## Usage
A wrapper is made available, see ``denoise-tmppca --help``.

```python
import nibabel
from tmppca import tmppca_cpp

data = nibabel.load('data.nii.gz')
mask = nibabel.load('mask.nii.gz').get_fdata().astype(bool)

denoised, sigma2, P, snr_gain = tmppca_cpp.denoise_tmppca(  data.get_fdata(),
                                                            window          = [5, 5, 5],
                                                            mask            = mask,
                                                            num_threads     = 8
                                                          )

nibabel.save(nibabel.Nifti1Image(denoised, data.affine, data.header), 'data_denoised.nii')
nibabel.save(nibabel.Nifti1Image(sigma2, data.affine, data.header), 'data_sigma2.nii')
```

### Standard data (regular MP-PCA with Olesen et al. noise estimation [2])

Concatenate all contrasts along the last axis, then pass them together:

```python
# data shape 4D -> 3D+Constrasts (Nx, Ny, Nz, N-contrast)
denoised, *_ = denoise_tmppca(data, window=[5, 5, 5])
```

### Multi-modal data (tensor MP-PCA)

For genuinely multi-dimensional measurement structures (e.g., window-voxels × TE × contrast; a.k.a., 3 "modes"), reshape your data accordingly (array shapes 3D+Contrast-1+Constrast-2).
Example:

```python
# data shape at least 5D -> 3D+Mode-contrast+Mode-TE (Nx, Ny, Nz, N-contrast, N-TE)
Nx, Ny, Nz  = data.shape[:3]
n_contrasts = 3
n_TE        = 6
data_5d     = data.reshape(Nx, Ny, Nz, n_TE, n_contrasts)

denoised, *_ = denoise_tmppca(data, window=[5, 5, 5])
```

---

## Input parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window` | — | Patch size per spatial dimension, e.g. `[5,5,5]` |
| `mask` | `None` | Binary brain mask |
| `stride` | `[1,1,1]` | Step size per spatial dimension (1 = full overlap of patches; higher is faster but coarsest) |
| `optim_shrinkage` | `True` | Gavish-Donoho optimal singular-value shrinkage |
| `joint_noise_est` | `True` | Pool noise estimates across all tensor modes |
| `num_threads` | `0` | OpenMP thread count (0 = all cores) |
| `center_only` | `False` | `True`: only restore the central voxel upon each patch denoising; `False`: each voxel is denoised N=Nx×Ny×Nz (window size) times & averaged |
| `mode_grouping` | auto | Controls how patch dimensions are grouped into tensor modes for the multi-level SVD. By default all spatial dims form one group and each measurement dim gets its own group (e.g. a 5D array with `window=[5,5,5]` gives `[[0,1,2],[3],[4]]`). Only set this explicitly for non-standard groupings such as fusing a spatial dimension with a measurement dimension. |

## Outputs

| Output |  Description |
|-----------|---------|
| `denoised` | Denoised image matrix |
| `sigma2`  | Noise variance map |
| `P` | Number of signal components (rank) retained after thresholding, per voxel |
| `snr_gain` | Theoretical factor by which the voxel-wise SNR improves after denoising |

