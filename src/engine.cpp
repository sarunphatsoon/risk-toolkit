#include "creditmc/engine.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <thread>

namespace creditmc {

double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0)); 
}

double norm_pdf(double x) {
    static constexpr double inv_sqrt_2pi = 0.39894228040143267794;
    return inv_sqrt_2pi * std::exp(-0.5 * x * x);
}

namespace {
double norm_inv_cdf_approx(double p){
    if (!(p > 0.0 && p < 1.0)) {
        throw std::invalid_argument("norm_inv_cdf: p must be in (0, 1)");
    }
        
    static constexpr double a[] = {-3.969683028665376e+01, 2.209460984245205e+02,
                                   -2.759285104469687e+02, 1.383577518672690e+02,
                                   -3.066479806614716e+01, 2.506628277459239e+00};
       
    static constexpr double b[] = {-5.447609879822406e+01, 1.615858368580409e+02,
                                   -1.556989798598866e+02, 6.680131188771972e+01,
                                   -1.328068155288572e+01};
    
    static constexpr double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                   -2.400758277161838e+00, -2.549732539343734e+00,
                                   4.374664141464968e+00,  2.938163982698783e+00};

    static constexpr double d[] = {7.784695709041462e-03, 3.224671290700398e-01,
                                   2.445134137142996e+00, 3.754408661907416e+00};

    static constexpr double p_low = 0.02425;
    static constexpr double p_high = 1.0 - p_low;
        
    if (p < p_low) {
        const double q = std::sqrt(-2.0 * std :: log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / 
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    
    if (p > p_high) {
        const double q = std::sqrt(-2.0 * std :: log (1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    const double q = p - 0.5;
    const double r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / 
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}
    
}
double norm_inv_cdf(double p) {
    if (!(p > 0.0 && p < 1.0)) {
        throw std::invalid_argument("norm_inv_cdf: p must be in (0, 1)");
    }
    if (p > 0.5) {
        return -norm_inv_cdf(1.0 - p);
    }
    double x = norm_inv_cdf_approx(p);
    const double e = norm_cdf(x) - p;
    const double u = e * 2.50662827463100050242 * std::exp(0.5 * x * x);
    x -= u / (1.0 + 0.5 * x * u);
    return x;
}

namespace {

std::uint64_t splitmix64(std::uint64_t x ) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
    
class Xoshiro256pp {
public:
    explicit Xoshiro256pp(std::uint64_t seed) {
        for (auto& word : s_) {
            seed += 0x9E3779B97F4A7C15ULL;
            word = splitmix64(seed);
        }
    }

    std::uint64_t next() {
        const std::uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
        const std::uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

    double uniform() {
        return static_cast<double>(next() >> 11) * 0x1.0p-53;
    }

private:
    static std::uint64_t rotl(std::uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

    std::uint64_t s_[4];
};

class NormalGenerator {
public:
    explicit NormalGenerator(std:: uint64_t seed) : rng_(seed) {}

    double operator()() {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        double u, v, s;
        do {
            u = 2.0 * rng_.uniform() - 1.0;
            v = 2.0 * rng_.uniform() - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);
        const double factor = std::sqrt(-2.0 * std:: log(s) / s);
        spare_ = v * factor;
        has_spare_ = true;
        return u * factor;
    }

private:
    Xoshiro256pp rng_;
    double spare_ = 0.0;
    bool has_spare_ = false;
};

void validate(const Portfolio& p, std::size_t n_scenarios) {
    const std::size_t n = p.pd.size();
    if (n == 0) {
        throw std::invalid_argument("portfolio is empty");
    }
    if (p.ead.size() != n || p.lgd.size() != n) {
        throw std::invalid_argument("pd, ead, and lgd must have the same length");
    }
    if (!(p.rho > 0.0 && p.rho < 1.0)) {
        throw std::invalid_argument("rho must be in (0,1)");
    }
    if (n_scenarios == 0){
        throw std::invalid_argument("n_scenarios must be positive");
    }
    for (std::size_t i = 0; i < n; ++i) {
        if (!(p.pd[i] > 0.0 && p.pd[i] < 1.0)) {
            throw std::invalid_argument("each pd must be in (0, 1)");
        }
        if (!(p.lgd[i] >= 0.0 && p.lgd[i] <= 1.0)) {
            throw std::invalid_argument("each lgd must be in [0, 1]");
        }
        if (!(p.ead[i] >= 0.0)) {
            throw std::invalid_argument("each ead must be >= 0");
        }
    }
}

}

std::vector<double> simulate_losses(const Portfolio& portfolio, 
                                    std::size_t n_scenarios,
                                    std::uint64_t seed,
                                    unsigned n_threads) {
    validate(portfolio, n_scenarios);
                                    
    const std::size_t n_obligors = portfolio.pd.size();

    std::vector<double> threshold(n_obligors);
    std::vector<double> loss_if_default(n_obligors);
    for (std::size_t i = 0; i < n_obligors; ++i) {
        threshold[i] = norm_inv_cdf(portfolio.pd[i]);
        loss_if_default[i] = portfolio.ead[i] * portfolio.lgd[i];
    }
    const double sqrt_rho = std::sqrt(portfolio.rho);
    const double sqrt_one_minus_rho = std::sqrt(1.0 - portfolio.rho);

    std::vector<double> losses(n_scenarios);

    constexpr std::size_t kBlockSize = 1024;
    const std::size_t n_blocks = (n_scenarios + kBlockSize - 1) / kBlockSize;

    if (n_threads == 0) {
        n_threads = std::max(1u, std::thread::hardware_concurrency());
    }
    n_threads = static_cast<unsigned>(std::min<std::size_t>(n_threads, n_blocks));

    std::atomic<std::size_t> next_block{0};

    auto worker = [&]() {
        for (;;) {
            const std::size_t block = next_block.fetch_add(1, std::memory_order_relaxed);
            if (block >= n_blocks) {
                return;
            }

            NormalGenerator normal(splitmix64(seed ^ splitmix64(block)));

            const std::size_t begin = block * kBlockSize;
            const std::size_t end = std::min(begin + kBlockSize, n_scenarios);

            for (std::size_t s = begin; s < end; ++s) {
                const double z = normal();
                double loss = 0.0;
                for (std::size_t i = 0; i < n_obligors; ++i) {
                    const double eps = normal();
                    const double asset = sqrt_rho * z + sqrt_one_minus_rho * eps;
                    if (asset < threshold[i]) {
                        loss += loss_if_default[i];
                    }
                }
                losses[s] = loss;
            }
        }
    };

    std::vector<std::thread> pool;
    for (unsigned t = 1; t < n_threads; ++t) {
        pool.emplace_back(worker);
    }
    worker();
    for (auto& th : pool) {
        th.join();
    }

    return losses;
}

double irb_asymptotic_loss(double pd, double lgd, double rho, double confidence) {
    const double conditional_pd = norm_cdf(
        (norm_inv_cdf(pd) + std::sqrt(rho) * norm_inv_cdf(confidence)) / std:: sqrt(1.0 - rho));
    return lgd * conditional_pd;

}

}

