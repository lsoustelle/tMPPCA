#pragma once

#include <Eigen/Dense>
#include <vector>
#include <numeric>
#include <cmath>
#include <cstring>
#include <limits>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// ============================================================
//  Internal types
// ============================================================
struct NoiseEstimate { double sigma2; int P; };
struct DiscardResult { Matrix U; Vector s; Matrix V; int P; };
struct DenoiseResult { Matrix X; double sigma2; std::vector<int> P; };

// ============================================================
//  estimate_noise
//
//  Symmetric MP-PCA noise estimator (Olesen et al. 2022).
//
//  Combines two independent sigma² estimates at each candidate rank p:
//    sigmasq_1[p]: energy-based — tail variance / remaining degrees of freedom
//    sigmasq_2[p]: width-based  — eigenvalue span / theoretical MP bulk width
//
//  The threshold t is the first p where the width-implied sigma² drops below
//  the energy-implied sigma². Requiring both estimators to agree simultaneously
//  makes the threshold more robust than the energy criterion alone, particularly
//  when signal and noise eigenvalues overlap near the MP spectral edge.
//
//  Parameters
//  ----------
//  s : singular values (descending), length r = min(M, N)
//  M : rows of the unfolded matrix
//  N : cols of the unfolded matrix (N >= M assumed; symmetric if not)
// ============================================================
inline NoiseEstimate estimate_noise(const Vector& s, int M, int N)
{
    const int r  = (int)s.size();       // = min(M, N)
    const int Mp = r;
    const int Np = (M > N) ? M : N;    // max(M, N)

    // ── Energy estimator (sigmasq_1) 
    // sigmasq_1[p] = sum(s²[p:]) / ((Mp-p) * (Np-p))
    // Iterate from the tail, accumulating the suffix sum.
    std::vector<double> sigmasq_1(Mp);
    {
        double tail = 0.0;
        for (int k = Mp - 1; k >= 0; --k) tail += s(k) * s(k);
        for (int p = 0; p < Mp; ++p)
        {
            sigmasq_1[p] = tail / ((double)(Mp - p) * (double)(Np - p));
            tail -= s(p) * s(p);
        }
    }

    // ── Width estimator (sigmasq_2) 
    // sigmasq_2[p] = (s²[p] - s²[Mp-1]) / (4 * sqrt(Np * Mp))
    // Theoretical MP bulk width = 4 * sigma² * sqrt(Np * Mp).
    // Defined for p in [0, Mp-2] only (need at least one bulk eigenvalue).
    const double range_mp  = 4.0 * std::sqrt((double)Np * (double)Mp);
    const double last_val2 = s(Mp - 1) * s(Mp - 1);

    // ── Find crossover 
    // t = first p in [0, Mp-2] where sigmasq_2[p] < sigmasq_1[p].
    int t = Mp - 2;  // fallback: use near-last component
    for (int p = 0; p < Mp - 1; ++p)
    {
        const double sigmasq_2_p = (s(p) * s(p) - last_val2) / range_mp;
        if (sigmasq_2_p < sigmasq_1[p])
        {
            t = p;
            break;
        }
    }

    double sigma2 = sigmasq_1[t];

    // Guard: degenerate patch (pure noise or numerical zero)
    if (sigma2 <= 0.0 || !std::isfinite(sigma2))
        return { std::numeric_limits<double>::epsilon(), 0 };

    return { sigma2, t };
}

// ============================================================
//  discard_noise
// ============================================================
inline DiscardResult discard_noise(
    const Matrix& U, const Vector& s, const Matrix& V, double sigma2)
{
    const int    M      = (int)U.rows();
    const int    N      = (int)V.rows();
    const double cutoff = sigma2 * (std::sqrt((double)M) + std::sqrt((double)N))
                                 * (std::sqrt((double)M) + std::sqrt((double)N));
    int P = 0;
    while (P < (int)s.size() && s(P) * s(P) > cutoff) ++P;

    if (P == 0) return { U.leftCols(1).eval(), s.head(1).eval(), V.leftCols(1).eval(), 0 };
    return { U.leftCols(P).eval(), s.head(P).eval(), V.leftCols(P).eval(), P };
}

// ============================================================
//  apply_optimal_shrinkage
//  Only called when P > 0 (caller's responsibility).
// ============================================================
inline Vector apply_optimal_shrinkage(const Vector& s, int M, int N, double sigma2)
{
    const double mn2   = (double)(N - M) * (double)(N - M);
    const double coeff = 2.0 * (double)(M + N) * sigma2;
    Vector out = s.array().square();
    out = (out.array() - coeff
           + mn2 * sigma2 * sigma2 / out.array().max(1e-12)
          ).max(0.0).sqrt();
    return out;
}

// ============================================================
//  denoise_array_tmppca
//
//  Single-patch tMPPCA denoiser.
//
//  Parameters
//  ----------
//  ptr             : Fortran-order (col-major) data pointer
//  dims            : tensor shape after mode grouping
//  optim_shrinkage : apply Gavish-Donoho optimal singular-value shrinkage
//  joint_noise_est : pool noise estimates from all modes (only meaningful
//                    when dims.size() >= 3; ignored for 2-mode data)
// ============================================================
inline DenoiseResult denoise_array_tmppca(
    const double*               ptr,
    const std::vector<ssize_t>& dims,
    bool                        optim_shrinkage,
    bool                        joint_noise_est)
{
    const int    nd          = (int)dims.size();
    const size_t total_elems = std::accumulate(
        dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());

    // --------------------------------------------------------
    // SIGMA ESTIMATION PASS
    // --------------------------------------------------------
    // joint_noise_est is only beneficial when nd >= 3.
    // For nd == 2 a single estimation SVD is both sufficient and correct.
    const int nSVD = (joint_noise_est && nd > 2) ? nd : 1;

    Matrix X_work = Eigen::Map<const Matrix>(ptr, dims[0], total_elems / dims[0]);

    struct ModeInfo { Vector s; ssize_t rows, cols; int P; };
    std::vector<ModeInfo> modes;
    modes.reserve(nSVD);

    Matrix n0_U, n0_V;
    Vector n0_s;

    for (int n = 0; n < nSVD; ++n)
    {
        const ssize_t rows = dims[n];
        const ssize_t cols = (ssize_t)X_work.size() / rows;
        Matrix Xc = Eigen::Map<const Matrix>(X_work.data(), rows, cols);

        if (n == 0)
        {
            Eigen::BDCSVD<Matrix> svd(Xc, Eigen::ComputeThinU | Eigen::ComputeThinV);
            n0_s = svd.singularValues().eval();
            n0_U = svd.matrixU().eval();
            n0_V = svd.matrixV().eval();

            NoiseEstimate ne = estimate_noise(n0_s, (int)rows, (int)cols);
            modes.push_back({ n0_s, rows, cols, ne.P });
            X_work.noalias() = n0_V * n0_s.asDiagonal();
        }
        else
        {
            Eigen::BDCSVD<Matrix> svd(Xc, Eigen::ComputeThinV);
            const Vector s = svd.singularValues().eval();
            const Matrix V = svd.matrixV().eval();

            NoiseEstimate ne = estimate_noise(s, (int)rows, (int)cols);
            modes.push_back({ s, rows, cols, ne.P });
            X_work.noalias() = V * s.asDiagonal();
        }
    }

    // Combined noise estimate
    double num = 0.0, denom = 0.0;
    for (const auto& m : modes)
    {
        for (int k = m.P; k < (int)m.s.size(); ++k) num += m.s(k) * m.s(k);
        denom += (double)(m.rows - m.P) * (double)(m.cols - m.P);
    }
    double sigma2 = (denom > 0.0) ? num / denom : 0.0;

    // Re-derive per-mode P from combined sigma2
    for (auto& m : modes)
    {
        const double edge   = std::sqrt((double)m.rows) + std::sqrt((double)m.cols);
        const double cutoff = sigma2 * edge * edge;
        m.P = 0;
        while (m.P < (int)m.s.size() && m.s(m.P) * m.s(m.P) > cutoff) ++m.P;
    }

    // --------------------------------------------------------
    // RECURSIVE SVD PASS
    // --------------------------------------------------------
    const int nd_rec = (nd <= 2) ? 1 : nd;

    std::vector<Matrix>  U_list;
    std::vector<int>     P_list;
    std::vector<ssize_t> cols_list;
    U_list.reserve(nd_rec);
    P_list.reserve(nd_rec);
    cols_list.reserve(nd_rec);

    X_work = Eigen::Map<const Matrix>(ptr, dims[0], total_elems / dims[0]);

    for (int n = 0; n < nd_rec; ++n)
    {
        const ssize_t rows = dims[n];
        const ssize_t cols = (ssize_t)X_work.size() / rows;
        cols_list.push_back(cols);

        DiscardResult dr;
        if (n == 0)
            dr = discard_noise(n0_U, n0_s, n0_V, sigma2);
        else
        {
            Matrix Xc = Eigen::Map<const Matrix>(X_work.data(), rows, cols);
            Eigen::BDCSVD<Matrix> svd(Xc, Eigen::ComputeThinU | Eigen::ComputeThinV);
            dr = discard_noise(svd.matrixU(), svd.singularValues(),
                               svd.matrixV(), sigma2);
        }

        if (optim_shrinkage && n == nd_rec - 1 && dr.P > 0)
            dr.s = apply_optimal_shrinkage(dr.s, (int)dr.U.rows(), (int)dr.V.rows(), sigma2);

        Matrix V_keep = dr.V;
        Vector s_keep = dr.s;

        U_list.push_back(std::move(dr.U));
        P_list.push_back(dr.P);

        if (dr.P == 0) break;

        X_work.noalias() = V_keep * s_keep.asDiagonal();
    }

    // --------------------------------------------------------
    // BACKWARD RECONSTRUCTION
    // --------------------------------------------------------
    const int num_levels = (int)U_list.size();

    for (int n = num_levels - 1; n >= 0; --n)
    {
        if (P_list[n] == 0)
            X_work = Matrix::Zero(dims[n], cols_list[n]);
        else
        {
            const int P = P_list[n];
            Matrix tmp = Eigen::Map<const Matrix>(X_work.data(), X_work.size() / P, P);
            X_work.noalias() = U_list[n] * tmp.transpose();
        }
    }

    P_list.resize(nd, 0);
    return { std::move(X_work), sigma2, std::move(P_list) };
}
