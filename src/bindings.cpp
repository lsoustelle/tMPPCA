#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "core_tmppca.hpp"
#include "denoise_tmppca.hpp"

namespace py = pybind11;

// ============================================================
//  denoise_array_tmppca  (single-patch)
// ============================================================
static py::tuple py_denoise(
    py::array_t<double, py::array::f_style | py::array::forcecast> X_in,
    bool optim_shrinkage,
    bool joint_noise_est)
{
    auto buf = X_in.request();
    if (buf.ndim < 2)
        throw std::runtime_error("Input must have at least 2 dimensions");

    const std::vector<ssize_t> dims(buf.shape.begin(), buf.shape.end());
    const int nd = (int)buf.ndim;
    const double* ptr = static_cast<const double*>(buf.ptr);

    DenoiseResult res = denoise_array_tmppca(
        ptr, dims, optim_shrinkage, joint_noise_est);

    // Build Fortran-order output array
    std::vector<ssize_t> strides(nd);
    ssize_t s = sizeof(double);
    for (int i = 0; i < nd; ++i) { strides[i] = s; s *= dims[i]; }

    py::array_t<double> out(dims, strides);
    std::memcpy(out.mutable_data(), res.X.data(), sizeof(double) * res.X.size());

    return py::make_tuple(out, res.sigma2, res.P);
}

// ============================================================
//  denoise_tmppca  (sliding-window)
// ============================================================
static py::tuple py_denoise_sliding(
    py::array_t<double, py::array::f_style | py::array::forcecast> data_in,
    py::array_t<int>               window_in,
    std::vector<std::vector<int>>  mode_grouping,
    py::object                     mask_obj,
    std::vector<int>               stride,
    bool                           center_only,
    bool                           optim_shrinkage,
    bool                           joint_noise_est,
    int                            num_threads)
{
    auto buf = data_in.request();
    const int nd = (int)buf.ndim;
    if (nd < 2) throw std::runtime_error("data must have at least 2 dimensions");

    auto win_buf = window_in.request();
    const int nw = (int)win_buf.size;
    if (nw >= nd) throw std::runtime_error("window length must be < data ndim");

    const int* win_ptr = static_cast<const int*>(win_buf.ptr);
    std::vector<int> window(win_ptr, win_ptr + nw);
    std::vector<int> dims_vox(buf.shape.begin(), buf.shape.begin() + nw);
    std::vector<int> dims_meas(buf.shape.begin() + nw, buf.shape.end());

    const int num_vox = (int)std::accumulate(
        dims_vox.begin(), dims_vox.end(), 1, std::multiplies<int>());
    const double* data_ptr = static_cast<const double*>(buf.ptr);

    // mask
    const bool* mask_ptr = nullptr;
    py::array_t<bool> mask_arr;
    if (!mask_obj.is_none())
    {
        mask_arr = mask_obj.cast<py::array_t<bool, py::array::f_style | py::array::forcecast>>();
        auto mb = mask_arr.request();
        if (mb.size != num_vox)
            throw std::runtime_error("mask size does not match spatial dimensions");
        mask_ptr = static_cast<const bool*>(mb.ptr);
    }

    SlidingWindowResult res = denoise_tmppca(
        data_ptr, dims_vox, dims_meas, window,
        mode_grouping, mask_ptr, stride, center_only,
        optim_shrinkage, joint_noise_est, num_threads);

    // Build Fortran-order output arrays
    std::vector<ssize_t> data_shape(buf.shape.begin(), buf.shape.end());
    std::vector<ssize_t> vox_shape(buf.shape.begin(), buf.shape.begin() + nw);

    auto f_strides = [](const std::vector<ssize_t>& shape) {
        std::vector<ssize_t> st(shape.size());
        ssize_t s = sizeof(double);
        for (int i = 0; i < (int)shape.size(); ++i) { st[i] = s; s *= shape[i]; }
        return st;
    };

    py::array_t<double> denoised_out(data_shape, f_strides(data_shape));
    std::memcpy(denoised_out.mutable_data(), res.denoised.data(),
                sizeof(double) * res.denoised.size());

    py::array_t<double> sigma2_out(vox_shape, f_strides(vox_shape));
    std::memcpy(sigma2_out.mutable_data(), res.sigma2.data(),
                sizeof(double) * res.sigma2.size());

    std::vector<ssize_t> p_shape(buf.shape.begin(), buf.shape.begin() + nw);
    p_shape.push_back(res.n_modes);
    py::array_t<double> p_out(p_shape, f_strides(p_shape));
    std::memcpy(p_out.mutable_data(), res.P.data(), sizeof(double) * res.P.size());

    py::array_t<double> snr_out(vox_shape, f_strides(vox_shape));
    std::memcpy(snr_out.mutable_data(), res.snr_gain.data(),
                sizeof(double) * res.snr_gain.size());

    return py::make_tuple(denoised_out, sigma2_out, p_out, snr_out);
}

// ============================================================
//  Module
// ============================================================
PYBIND11_MODULE(tmppca_cpp, m)
{
    m.doc() = "Recursive tensor MPPCA denoiser (C++ / Eigen backend)";

    m.def("denoise_array_tmppca",
          &py_denoise,
          py::arg("X"),
          py::arg("optim_shrinkage")  = false,
          py::arg("joint_noise_est")  = true,
          R"doc(
Denoise a single patch (inner denoiser).

Parameters
----------
X               : ndarray, Fortran-order, shape (mode0, mode1, ...)
optim_shrinkage : optimal singular-value shrinkage (default False)
joint_noise_est : use all modes for noise estimation (default True)

Returns
-------
(denoised, sigma2, P)
          )doc");

    m.def("denoise_tmppca",
          &py_denoise_sliding,
          py::arg("data"),
          py::arg("window"),
          py::arg("mode_grouping")     = std::vector<std::vector<int>>{},
          py::arg("mask")              = py::none(),
          py::arg("stride")            = std::vector<int>{},
          py::arg("center_only")       = false,
          py::arg("optim_shrinkage")   = true,
          py::arg("joint_noise_est")   = true,
          py::arg("num_threads")       = 0,
          R"doc(
Sliding-window recursive tensor MPPCA denoiser.

Parameters
----------
data            : ndarray, shape (*spatial, *measurements), Fortran order preferred
window          : int array, patch size per spatial dim
mode_grouping   : list of lists of int — tensor grouping (0-based). Default: group all spatial dims.
mask            : bool array, shape = spatial dims (optional)
stride          : int array, step per spatial dim (default: all 1s)
center_only     : only accumulate centre voxel per patch (default False)
optim_shrinkage : optimal singular-value shrinkage (default False)
joint_noise_est : use all modes for noise estimation (default True)
num_threads     : OpenMP thread count, 0 = use OMP_NUM_THREADS / all cores (default 0)

Returns
-------
(denoised, sigma2, P, snr_gain)
          )doc");
}