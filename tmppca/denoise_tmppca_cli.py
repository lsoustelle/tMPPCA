import warnings
import sys
import os
import time
import argparse; from argparse import RawTextHelpFormatter
import subprocess

import numpy
import nibabel

try:
    from tmppca import tmppca_cpp
except ImportError:
    raise ImportError(
        'Could not import tmppca_cpp extension. '
        'Please rebuild the package: pip install .'
    )

def main():
    ## parse arguments
    text_description = "Sliding-window (tensor) MP-PCA for multi-dimensional MRI data denoising.\n \
        Implements the MP-PCA algorithm with symmetric noise estimator as in [2].\n \
        The denoiser operates on 4D (X,Y,Z,N; regular MP-PCA) or 5D (X,Y,Z,echoes,contrasts; tensor MP-PCA) data.\n \
        \nReferences: \
            \n\t [1] Veraart J et al., Denoising of diffusion MRI using random matrix theory, NeuroImage 2016. \
            \n\t [2] Olesen JL et al., Tensor denoising of multidimensional MRI data, Magn Reson Med. 2022. \
            \n\t [3] Fernandes F et al., MP-PCA denoising of fMRI time-series data can lead to artificial activation “spreading”, NeuroImage 2023. \
        \nPlease cite: Olesen JL et al., Magn Reson Med 2022. doi:10.1002/mrm.29478"
    
    parser = argparse.ArgumentParser(description=text_description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('input',        help="Input NIfTI path (4D or 5D image)")
    parser.add_argument('output',       help="Output denoised NIfTI path")
    parser.add_argument('--sigma2',     nargs="?", help="Output sigma² map NIfTI path (optional)")
    parser.add_argument('--snr_gain',   nargs="?", help="Output SNR gain map NIfTI path (optional)")
    parser.add_argument('--window',     required=True, nargs="?", default="5,5,5",
                                        help="Patch window size (comma-separated), must match spatial dims of input.\n"
                                             "  - For 3D spatial data: 3 values (e.g. 5,5,5)\n"
                                             "  - For 2D spatial data: 2 values (e.g. 5,5)\n"
                                             "  (default: 5,5,5)")
    parser.add_argument('--mask',       nargs="?", help="Input binary brain mask NIfTI path (advised)")
    parser.add_argument('--num_threads', nargs="?", type=int, default=0,
                                        help="Number of OpenMP threads (default: 0 = all cores).")

    ## denoising parameters
    parser.add_argument('--stride',     nargs="?", default=None,
                                        help="Sliding window stride (comma-separated, one value per spatial dim).\n"
                                             "Stride > 1 trades denoising quality for speed.\n"
                                             "  e.g. --stride 2,2,2  (default: 1,1,1)")
    parser.add_argument('--center_only', action='store_true',
                                        help="Denoise only the centre voxel of each patch (faster, less\n"
                                             "smoothing across overlapping patches; equivalent to no averaging).")
    parser.add_argument('--no_shrinkage', action='store_true',
                                        help="Disable Gavish-Donoho optimal singular value shrinkage\n"
                                             "(plain hard-threshold instead; not recommended).")
    parser.add_argument('--no_joint_noise', action='store_true',
                                        help="Disable joint noise estimation across tensor modes.\n"
                                             "Only meaningful for 5D data (nd >= 3); ignored otherwise.")
    parser.add_argument('--mode_grouping', nargs="?", default=None,
                                        help="Tensor mode grouping for 5D data (semicolon-separated groups of\n"
                                             "comma-separated 0-based axis indices).\n"
                                             "Auto-derived from window and data ndim if not provided.\n"
                                             "  e.g. for 5D (X,Y,Z,echoes,contrasts) with window=[5,5,5]:\n"
                                             "  --mode_grouping '0,1,2;3;4' (do not omit the quotes!)\n"
                                             "  -> [[0,1,2],[3],[4]]")

    args = parser.parse_args()

    #### parse & cast arguments
    input_niipath   = args.input
    output_niipath  = args.output
    sigma2_niipath  = args.sigma2
    snrgain_niipath = args.snr_gain
    mask_niipath    = args.mask
    num_threads     = args.num_threads if args.num_threads is not None else 0
    num_threads     = get_physCPU_number() if num_threads > get_physCPU_number() else num_threads
    optim_shrinkage = not args.no_shrinkage
    joint_noise_est = not args.no_joint_noise
    center_only     = args.center_only

    #### parse window
    try:
        window = [int(w) for w in args.window.split(',')]
    except ValueError:
        parser.error('--window must be a comma-separated list of integers (e.g. 5,5,5)')
    nw = len(window)

    #### parse stride
    stride = None
    if args.stride is not None:
        try:
            stride = [int(s) for s in args.stride.split(',')]
        except ValueError:
            parser.error('--stride must be a comma-separated list of integers (e.g. 2,2,2)')

    #### parse mode_grouping
    mode_grouping = None
    if args.mode_grouping is not None:
        try:
            mode_grouping = [[int(i) for i in grp.split(',')]
                             for grp in args.mode_grouping.split(';')]
        except ValueError:
            parser.error('--mode_grouping must be semicolon-separated groups of comma-separated integers (e.g. ''0,1,2;3;4'')')

    ####
    print('')
    print('--------------------------------------------------')
    print('---- Checking entries for (t)MP-PCA denoising ----')
    print('--------------------------------------------------')

    #### Load input
    if not os.path.isfile(input_niipath):
        parser.error('Input file not found: {}'.format(input_niipath))
    print('Input image: {}'.format(input_niipath))
    input_NII   = nibabel.load(input_niipath)
    input_data  = input_NII.get_fdata().astype(numpy.float64)
    input_shape = input_data.shape
    nd          = input_data.ndim
    print('  shape\t\t -> {}'.format(input_shape))
    print('  ndim\t\t -> {}'.format(nd))

    ## data must be at least 4D
    if nd < 4:
        parser.error('Input image must be at least 4D (found {}D). A 3D image has no measurement dimension to denoise.'.format(nd))
    if nd > 5:
        parser.error('Input image must be 4D or 5D (found {}D).'.format(nd))

    #### Check window vs spatial dims
    if nw not in (2, 3):
        parser.error('--window must have 2 or 3 values (found {}).'.format(nw))
    if nw != nd - (nd - 3):   # nw must equal number of spatial dims (nd-1 for 4D, nd-2 for 5D)
        # spatial dims = first (nd - n_meas_dims) dims
        # for 4D: 3 spatial; for 5D: 3 spatial -> always expect nw=3 unless data is 2D+meas
        pass # checked more carefully below

    ## determine number of spatial dims from data ndim and window size
    ## convention: window defines spatial dims, remaining are measurement dims
    n_spatial = nw
    if nd == 4 and nw == 2:
        # 2D+1 spatial + 1 measurement: valid (e.g. slice-by-slice)
        pass
    elif nd == 4 and nw == 3:
        # standard 3D+1 measurement
        pass
    elif nd == 5 and nw == 3:
        # standard 3D + 2 measurement dims (echoes x contrasts)
        pass
    elif nd == 5 and nw == 2:
        parser.error('For 5D data a 3-value window is required (--window x,y,z). Found 2-value window with 5D input.')
    else:
        parser.error('Unsupported combination: {}D data with {}-value window.'.format(nd, nw))

    ## each window dimension must be odd and must not exceed image extent
    for i, w in enumerate(window):
        if w % 2 == 0:
            parser.error('--window values must be odd integers (found even value {} at position {}).'.format(w, i))
        if w < 1:
            parser.error('--window values must be >= 1 (found {} at position {}).'.format(w, i))
        if w > input_shape[i]:
            parser.error('--window value {} at position {} exceeds image dimension {} (image size: {}).'.format(w, i, i, input_shape[i]))

    #### Check stride vs window
    if stride is not None:
        if len(stride) != nw:
            parser.error('--stride must have the same number of values as --window '
                         '(expected {}, found {}).'.format(nw, len(stride)))
        for i, s in enumerate(stride):
            if s < 1:
                parser.error('--stride values must be >= 1 (found {} at position {}).'.format(s, i))
            if s > window[i]:
                parser.error('--stride value {} at position {} exceeds window size {} - '
                             'this would leave unvisited voxels.'.format(s, i, window[i]))

    #### Check mode_grouping vs data ndim
    if mode_grouping is not None:
        array_ndim = nw + (nd - nw)  # = nd
        all_indices = [idx for grp in mode_grouping for idx in grp]
        ## all indices must be in [0, nd-1]
        for idx in all_indices:
            if idx < 0 or idx >= array_ndim:
                parser.error('--mode_grouping index {} out of range '
                             '(valid range: 0 to {}).'.format(idx, array_ndim - 1))
        ## no duplicates
        if len(all_indices) != len(set(all_indices)):
            parser.error('--mode_grouping contains duplicate axis indices.')
        ## spatial dims (0..nw-1) must all appear in exactly one group
        spatial_covered = set(range(nw))
        grouping_flat   = set(all_indices)
        if not spatial_covered.issubset(grouping_flat):
            missing = spatial_covered - grouping_flat
            parser.error('--mode_grouping is missing spatial axis indices: {}'.format(sorted(missing)))
    else:
        if nd == 5:
            ## auto-derive and inform user
            mode_grouping_info = [list(range(nw))] + [[nw + k] for k in range(nd - nw)]
            print('mode_grouping auto-derived -> {} '
                  '(use --mode_grouping to override).'.format(mode_grouping_info))
            print('')

    ## MP-PCA well-posed if n_win<n_meas
    n_win   = int(numpy.prod(window))
    n_meas  = int(numpy.prod(input_shape[nw:]))
    print('  patch matrix\t -> ({} spatial voxels) x ({} measurements)'.format(n_win, n_meas)) 

    #### Load mask
    mask_data = None
    if mask_niipath is not None:
        if not os.path.isfile(mask_niipath):
            parser.error('Mask file not found: {}'.format(mask_niipath))
        print('Mask image: {}'.format(mask_niipath))
        mask_NII  = nibabel.load(mask_niipath)
        mask_data = mask_NII.get_fdata().astype(numpy.float64)
        ## mask must match spatial dims of input
        if mask_data.shape != input_shape[:nw]:
            parser.error('Mask spatial shape {} does not match input spatial shape {} '
                         '(first {} dims).'.format(mask_data.shape, input_shape[:nw], nw))
        print('  shape\t\t -> {}'.format(mask_data.shape))
        print('  nonzero voxels -> {}'.format(int(mask_data.sum())))
        print('')

    #### Check output path is writable
    out_dir = os.path.dirname(os.path.abspath(output_niipath))
    if not os.path.isdir(out_dir):
        parser.error('Output directory does not exist: {}'.format(out_dir))

    ####
    print('--------------------------------------------------')
    print('---- (t)MP-PCA denoising parameters --------------')
    print('--------------------------------------------------')
    print('  Window size\t\t: {}'.format(window))
    print('  Stride size\t\t: {}'.format(stride if stride is not None else [1]*nw))
    print('  Center only\t\t: {}'.format(center_only))
    print('  Optimal shrinkage\t: {}'.format(optim_shrinkage))
    print('  Joint noise estim.\t: {}'.format(joint_noise_est))
    if mode_grouping is not None:
        print('  Mode grouping\t\t: {}'.format(mode_grouping))
    print('  Number of threads\t: {}'.format(num_threads if num_threads > 0 else "all cores"))
    print('')

    ####
    print('--------------------------------------------------')
    print('---- Running (t)MP-PCA denoising -----------------')
    print('--------------------------------------------------')
    start_time = time.time()
    w    = numpy.array(window, dtype=numpy.int32)
    mg   = mode_grouping if mode_grouping is not None else []
    st   = stride        if stride        is not None else []
    denoised, sigma2_map, _, snr_gain_map = tmppca_cpp.denoise_tmppca(
        input_data, w, mg, mask_data, st,
        center_only, optim_shrinkage, joint_noise_est, num_threads)
    print('---- Done in {:.3f} seconds ----'.format(time.time() - start_time))
    print('')

    ####
    def _save_nii(data, path):
        img = nibabel.Nifti1Image(data.astype(numpy.float32), input_NII.affine, input_NII.header)
        nibabel.save(img, path)

    _save_nii(denoised,     output_niipath)
    if sigma2_niipath  is not None:
        _save_nii(sigma2_map,   sigma2_niipath)
    if snrgain_niipath is not None:
        _save_nii(snr_gain_map, snrgain_niipath)
    print('')

###################################################################
############## Get CPU info
################################################################### 
def get_physCPU_number():
    # from joblib source code (commit d5c8274)
    # https://github.com/joblib/joblib/blob/master/joblib/externals/loky/backend/context.py#L220-L246
    if sys.platform == "linux":
        cpu_info = subprocess.run(
            "lscpu --parse=core".split(" "), capture_output=True)
        cpu_info = cpu_info.stdout.decode("utf-8").splitlines()
        cpu_info = {line for line in cpu_info if not line.startswith("#")}
        cpu_count_physical = len(cpu_info)
    elif sys.platform == "win32":
        cpu_info = subprocess.run(
            "wmic CPU Get NumberOfCores /Format:csv".split(" "),
            capture_output=True)
        cpu_info = cpu_info.stdout.decode('utf-8').splitlines()
        cpu_info = [l.split(",")[1] for l in cpu_info
                    if (l and l != "Node,NumberOfCores")]
        cpu_count_physical = sum(map(int, cpu_info))
    elif sys.platform == "darwin":
        cpu_info = subprocess.run(
            "sysctl -n hw.physicalcpu".split(" "), capture_output=True)
        cpu_info = cpu_info.stdout.decode('utf-8')
        cpu_count_physical = int(cpu_info)
    else:
        raise NotImplementedError(
            "unsupported platform: {}".format(sys.platform))
    if cpu_count_physical < 1:
            raise ValueError(
                "found {} physical cores < 1".format(cpu_count_physical))
    return cpu_count_physical

#### main
if __name__ == "__main__":
    sys.exit(main())