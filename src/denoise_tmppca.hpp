#pragma once

#include "core_tmppca.hpp"

#include <vector>
#include <cstring>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <atomic>
#include <cstdio>
#include <csignal>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================
//  SlidingWindowResult
// ============================================================
struct SlidingWindowResult
{
    std::vector<double> denoised;   // Fortran-order, same shape as input
    std::vector<double> sigma2;     // per-voxel, length num_vox
    std::vector<double> P;          // per-voxel × n_modes, length num_vox * n_modes
    std::vector<double> snr_gain;   // per-voxel, length num_vox
    int                 n_modes;
};

// ============================================================
//  Helper: flat Fortran-order index -> multi-index
// ============================================================
static inline void unravel_f(int idx, const std::vector<int>& dims, std::vector<int>& out)
{
    for (int d = 0; d < (int)dims.size(); ++d) { out[d] = idx % dims[d]; idx /= dims[d]; }
}

// ============================================================
//  denoise_tmppca (C++ sliding-window core)
//
//  Parameters
//  ----------
//  data_ptr        : Fortran-order input, shape (*dims_vox, *dims_meas)
//  dims_vox        : spatial dimensions
//  dims_meas       : measurement dimensions
//  window          : patch size per spatial dim
//  mode_grouping   : tensor grouping (0-based). Empty -> default.
//  mask_ptr        : bool mask, length num_vox, Fortran order. nullptr = all true.
//  stride          : step per spatial dim. Empty -> all ones.
//  center_only     : only accumulate centre voxel per patch
//  optim_shrinkage : optimal singular-value shrinkage
//  joint_noise_est : use all modes for noise estimation
//  num_threads     : OpenMP thread count (0 -> use OMP_NUM_THREADS / all cores)
// ============================================================
inline SlidingWindowResult denoise_tmppca(
    const double*                 data_ptr,
    const std::vector<int>&       dims_vox,
    const std::vector<int>&       dims_meas,
    const std::vector<int>&       window,
    std::vector<std::vector<int>> mode_grouping,
    const bool*                   mask_ptr,
    std::vector<int>              stride,
    bool                          center_only,
    bool                          optim_shrinkage,
    bool                          joint_noise_est,
    int                           num_threads)
{
    const int nw       = (int)dims_vox.size();
    const int nm       = (int)dims_meas.size();
    const int num_vox  = (int)std::accumulate(
        dims_vox.begin(), dims_vox.end(), 1, std::multiplies<int>());
    const int num_meas = (int)std::accumulate(
        dims_meas.begin(), dims_meas.end(), 1, std::multiplies<int>());

    if (stride.empty()) stride.assign(nw, 1);

    // ── patch tensor structure 
    std::vector<int> array_size;
    array_size.insert(array_size.end(), window.begin(), window.end());
    array_size.insert(array_size.end(), dims_meas.begin(), dims_meas.end());
    const int array_ndim = (int)array_size.size();

    if (mode_grouping.empty())
    {
        // Auto-derive: all spatial dims grouped, each measurement dim separate.
        // E.g. nw=3, nm=2 -> [[0,1,2],[3],[4]]
        std::vector<int> grp0;
        for (int k = 0; k < nw; ++k) grp0.push_back(k);
        mode_grouping.push_back(grp0);
        for (int k = 0; k < nm; ++k) mode_grouping.push_back({ nw + k });
    }

    std::vector<int> permute_order;
    for (auto& grp : mode_grouping) for (int d : grp) permute_order.push_back(d);
    for (int d = 0; d < array_ndim; ++d)
        if (std::find(permute_order.begin(), permute_order.end(), d) == permute_order.end())
            permute_order.push_back(d);

    std::vector<int> inv_permute(array_ndim);
    for (int i = 0; i < array_ndim; ++i) inv_permute[permute_order[i]] = i;

    std::vector<ssize_t> new_size_full;
    for (auto& grp : mode_grouping)
    {
        int sz = 1;
        for (int d : grp) sz *= array_size[d];
        new_size_full.push_back(sz);
    }
    {
        int covered = 0;
        for (auto& g : mode_grouping) covered += (int)g.size();
        if (covered < array_ndim)
        {
            int sz = 1;
            for (int k = covered; k < (int)permute_order.size(); ++k)
                sz *= array_size[permute_order[k]];
            if (sz > 1) new_size_full.push_back(sz);
        }
    }
    std::vector<ssize_t> new_size;
    for (auto s : new_size_full) if (s > 1) new_size.push_back(s);
    const int n_modes = (int)new_size.size();

    // new_size_snr = new_size + [1]
    std::vector<double> new_size_snr(new_size.begin(), new_size.end());
    new_size_snr.push_back(1.0);
    const double prod_ns = std::accumulate(
        new_size_snr.begin(), new_size_snr.end(), 1.0, std::multiplies<double>());

    // ── window index offsets 
    const int n_win = (int)std::accumulate(
        window.begin(), window.end(), 1, std::multiplies<int>());
    std::vector<int> increments(n_win);
    {
        std::vector<int> win_idx(nw);
        for (int k = 0; k < n_win; ++k)
        {
            int tmp = k;
            for (int d = 0; d < nw; ++d) { win_idx[d] = tmp % window[d]; tmp /= window[d]; }
            int inc = 0, sv = 1;
            for (int d = 0; d < nw; ++d) { inc += win_idx[d] * sv; sv *= dims_vox[d]; }
            increments[k] = inc;
        }
    }

    int centre_idx = 0;
    if (center_only)
    {
        int sv = 1;
        for (int d = 0; d < nw; ++d) { centre_idx += (window[d] / 2) * sv; sv *= window[d]; }
    }

    // permuted array_size (shape after permute)
    std::vector<int> array_size_perm(array_ndim);
    for (int i = 0; i < array_ndim; ++i) array_size_perm[i] = array_size[permute_order[i]];

    const int n_total = (int)std::accumulate(
        new_size.begin(), new_size.end(), 1, std::multiplies<ssize_t>());

    // ── identity-permute fast path 
    // For the common case (default mode_grouping, 1 measurement dim), permute_order
    // is [0,1,2,...] — the permute/inverse-permute loops are pure overhead.
    // Detect this once and skip those loops entirely inside the hot loop.
    bool is_identity_permute = true;
    for (int i = 0; i < array_ndim; ++i)
        if (permute_order[i] != i) { is_identity_permute = false; break; }

    // ── pre-allocate outputs 
    std::vector<double> denoised_flat(num_vox * (size_t)num_meas, 0.0);
    std::vector<double> count_v(num_vox, 0.0);
    std::vector<double> sigma2_flat(num_vox, 0.0);
    std::vector<double> P_flat(num_vox * (size_t)n_modes, 0.0);

    // ── OpenMP thread count 
#ifdef _OPENMP
    if (num_threads > 0) omp_set_num_threads(num_threads);
#endif

    // ── progress counter 
    // Pre-count eligible patches so the percentage is accurate.
    int num_patches = 0;
    {
        std::vector<int> iv(nw);
        for (int i = 0; i < num_vox; ++i)
        {
            unravel_f(i, dims_vox, iv);
            bool oob = false;
            for (int d = 0; d < nw && !oob; ++d) oob = (iv[d] + window[d] > dims_vox[d]);
            if (oob) continue;
            bool off = false;
            for (int d = 0; d < nw && !off; ++d) off = (iv[d] % stride[d] != 0);
            if (off) continue;
            if (mask_ptr)
            {
                bool any = false;
                for (int k = 0; k < n_win && !any; ++k) any = mask_ptr[i + increments[k]];
                if (!any) continue;
            }
            if (center_only && mask_ptr && !mask_ptr[i + increments[centre_idx]]) continue;
            ++num_patches;
        }
    }
    if (num_patches == 0) num_patches = 1;  // guard against division by zero
    std::atomic<int> patches_done{0};
    std::atomic<int> last_pct_printed{-1};

    // ── SIGINT handling 
    // CTRL+C sets this flag; the loop checks it and stops early.
    static std::atomic<int> sigint_caught{0};
    sigint_caught.store(0, std::memory_order_relaxed);
    auto prev_handler = std::signal(SIGINT, [](int){ sigint_caught.store(1, std::memory_order_relaxed); });

    // ── main loop 
    // raw_patch and idx_vec are declared firstprivate / thread-local so each
    // thread has its own pre-allocated buffer — no per-iteration heap allocs.
    std::vector<double> raw_patch_proto(n_total);   // template for firstprivate
    std::vector<int>    idx_vec(nw);

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 64) default(shared) \
        firstprivate(idx_vec, raw_patch_proto)
#endif
    for (int i = 0; i < num_vox; ++i)
    {
        if (sigint_caught.load(std::memory_order_relaxed)) continue;

        unravel_f(i, dims_vox, idx_vec);

        bool oob = false;
        for (int d = 0; d < nw && !oob; ++d) oob = (idx_vec[d] + window[d] > dims_vox[d]);
        if (oob) continue;

        bool off_stride = false;
        for (int d = 0; d < nw && !off_stride; ++d) off_stride = (idx_vec[d] % stride[d] != 0);
        if (off_stride) continue;

        if (mask_ptr)
        {
            bool any = false;
            for (int k = 0; k < n_win && !any; ++k)
                any = mask_ptr[i + increments[k]];
            if (!any) continue;
        }
        if (center_only && mask_ptr && !mask_ptr[i + increments[centre_idx]]) continue;

        // Reuse thread-private buffer (no heap alloc per patch)
        std::vector<double>& raw_patch = raw_patch_proto;

        // Step 1: gather — col-major (vox-major) layout
        for (int k = 0; k < n_win; ++k)
        {
            const int vox = i + increments[k];
            for (int m = 0; m < num_meas; ++m)
                raw_patch[k + (size_t)n_win * m] = data_ptr[vox + (size_t)num_vox * m];
        }

        // Step 2: permute raw_patch -> patch matrix
        // Fast path: identity permute -> direct map (zero-copy)
        Matrix patch(new_size[0], n_total / (int)new_size[0]);
        if (is_identity_permute)
        {
            std::memcpy(patch.data(), raw_patch.data(), sizeof(double) * n_total);
        }
        else
        {
            double* pp = patch.data();
            std::vector<int> orig_idx(array_ndim, 0);
            for (int orig_flat = 0; orig_flat < n_total; ++orig_flat)
            {
                int perm_flat = 0, sp = 1;
                for (int pd = 0; pd < array_ndim; ++pd)
                {
                    perm_flat += orig_idx[permute_order[pd]] * sp;
                    sp        *= array_size_perm[pd];
                }
                pp[perm_flat] = raw_patch[orig_flat];
                for (int d = 0; d < array_ndim; ++d)
                    if (++orig_idx[d] < array_size[d]) break; else orig_idx[d] = 0;
            }
        }

        // Denoise
        DenoiseResult res = denoise_array_tmppca(
            patch.data(), new_size, optim_shrinkage, joint_noise_est);

        // Progress: any thread increments; first to reach a new % prints it
        {
            int n   = ++patches_done;
            int pct = (100 * n) / num_patches;
            int old = last_pct_printed.load(std::memory_order_relaxed);
            if (pct > old &&
                last_pct_printed.compare_exchange_weak(
                    old, pct,
                    std::memory_order_relaxed,
                    std::memory_order_relaxed))
            {
                std::fprintf(stderr, "\r  -------- tMPPCA denoising: %3d%% --------", pct);
                std::fflush(stderr);
            }
        }

        // Step 4: inverse permute result -> raw_patch
        if (is_identity_permute)
        {
            std::memcpy(raw_patch.data(), res.X.data(), sizeof(double) * n_total);
        }
        else
        {
            const double* rp = res.X.data();
            std::vector<int> orig_idx(array_ndim, 0);
            for (int orig_flat = 0; orig_flat < n_total; ++orig_flat)
            {
                int perm_flat = 0, sp = 1;
                for (int pd = 0; pd < array_ndim; ++pd)
                {
                    perm_flat += orig_idx[permute_order[pd]] * sp;
                    sp        *= array_size_perm[pd];
                }
                raw_patch[orig_flat] = rp[perm_flat];
                for (int d = 0; d < array_ndim; ++d)
                    if (++orig_idx[d] < array_size[d]) break; else orig_idx[d] = 0;
            }
        }

        // Step 5: scatter with per-voxel atomics
        for (int k = 0; k < n_win; ++k)
        {
            if (center_only && k != centre_idx) continue;
            const int vox = i + increments[k];
            for (int m = 0; m < num_meas; ++m)
            {
#ifdef _OPENMP
                #pragma omp atomic
#endif
                denoised_flat[vox + (size_t)num_vox * m] += raw_patch[k + (size_t)n_win * m];
            }
        }

        const auto& p = res.P;
        if (center_only)
        {
            const int ci = i + increments[centre_idx];
#ifdef _OPENMP
            #pragma omp atomic
#endif
            count_v[ci] += 1.0;
#ifdef _OPENMP
            #pragma omp atomic
#endif
            sigma2_flat[ci] += res.sigma2;
            for (int m = 0; m < (int)p.size() && m < n_modes; ++m)
            {
#ifdef _OPENMP
                #pragma omp atomic
#endif
                P_flat[ci + (size_t)num_vox * m] += (double)p[m];
            }
        }
        else
        {
            for (int k = 0; k < n_win; ++k)
            {
                const int vi = i + increments[k];
#ifdef _OPENMP
                #pragma omp atomic
#endif
                count_v[vi] += 1.0;
#ifdef _OPENMP
                #pragma omp atomic
#endif
                sigma2_flat[vi] += res.sigma2;
                for (int m = 0; m < (int)p.size() && m < n_modes; ++m)
                {
#ifdef _OPENMP
                    #pragma omp atomic
#endif
                    P_flat[vi + (size_t)num_vox * m] += (double)p[m];
                }
            }
        }
    }
    // ── restore previous SIGINT handler
    std::signal(SIGINT, prev_handler);
    if (sigint_caught.load(std::memory_order_relaxed))
    {
        std::fprintf(stderr, "\r  -------- tMPPCA denoising: interrupted --------\n");
        std::raise(SIGINT);   // re-raise so Python sees KeyboardInterrupt
        return {};            // unreachable, keeps compiler happy
    }
    std::fprintf(stderr, "\r  -------- tMPPCA denoising: 100%% --------\n");

    // ── handle unvisited voxels
    for (int i = 0; i < num_vox; ++i)
    {
        if (count_v[i] == 0.0)
        {
            for (int m = 0; m < num_meas; ++m)
                denoised_flat[i + (size_t)num_vox * m] = data_ptr[i + (size_t)num_vox * m];
            sigma2_flat[i] = std::numeric_limits<double>::quiet_NaN();
            for (int m = 0; m < n_modes; ++m)
                P_flat[i + (size_t)num_vox * m] = std::numeric_limits<double>::quiet_NaN();
            count_v[i] = 1.0;
        }
    }

    // ── average 
    for (int i = 0; i < num_vox; ++i)
    {
        const double c = count_v[i];
        sigma2_flat[i] /= c;
        for (int m = 0; m < num_meas; ++m) denoised_flat[i + (size_t)num_vox * m] /= c;
        for (int m = 0; m < n_modes;  ++m) P_flat[i + (size_t)num_vox * m] /= c;
    }

    // ── SNR gain 
    std::vector<double> snr_gain(num_vox);
    for (int i = 0; i < num_vox; ++i)
    {
        if (std::isnan(sigma2_flat[i]))
            { snr_gain[i] = std::numeric_limits<double>::quiet_NaN(); continue; }

        double denom = 0.0;
        if (n_modes == 1)
        {
            double pv = P_flat[i];
            denom = pv * pv;
            for (int m = 0; m < (int)new_size_snr.size() - 1; ++m)
                denom += (new_size_snr[m] - pv) * pv;
        }
        else
        {
            double prod_p = 1.0;
            for (int m = 0; m < n_modes; ++m) prod_p *= P_flat[i + (size_t)num_vox * m];
            denom = prod_p;
            for (int m = 0; m < n_modes && m < (int)new_size_snr.size() - 1; ++m)
            {
                double pv = P_flat[i + (size_t)num_vox * m];
                denom += (new_size_snr[m] - pv) * pv;
            }
        }
        snr_gain[i] = (denom > 0.0) ? std::sqrt(prod_ns / denom)
                                     : std::numeric_limits<double>::quiet_NaN();
    }

    return { std::move(denoised_flat), std::move(sigma2_flat),
             std::move(P_flat), std::move(snr_gain), n_modes };
}