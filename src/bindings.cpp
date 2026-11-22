#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include <string>

#include "creditmc/backtest.hpp"
#include "creditmc/engine.hpp"
#include "creditmc/market_risk.hpp"

namespace py = pybind11;

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

namespace {

std::vector<double> to_vector(const DoubleArray& arr, const char* name) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must be a 1-D array");
    }
    const double* data = arr.data();
    return std::vector<double>(data, data + arr.size());
}

std::vector<int> to_int_vector(const IntArray& arr, const char* name) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must be a 1-D array");
    }
    const int* data = arr.data();
    return std::vector<int>(data, data + arr.size());
}

py::array_t<double> to_array(const std::vector<double>& values) {
    py::array_t<double> result(static_cast<py::ssize_t>(values.size()));
    if (!values.empty()) {
        std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(double));
    }
    return result;
}

py::array_t<double> simulate(const DoubleArray& pd, const DoubleArray& ead, const DoubleArray& lgd,
                             double rho, std::size_t n_scenarios, std::uint64_t seed,
                             unsigned n_threads) {
    creditmc::Portfolio portfolio{to_vector(pd, "pd"), to_vector(ead, "ead"),
                                  to_vector(lgd, "lgd"), rho};

    std::vector<double> losses;
    {
        // Release the GIL so Python threads are not blocked during the simulation.
        py::gil_scoped_release release;
        losses = creditmc::simulate_losses(portfolio, n_scenarios, seed, n_threads);
    }
    return to_array(losses);
}

py::tuple historical_var_es(const DoubleArray& returns, std::size_t window, double alpha) {
    const std::vector<double> r = to_vector(returns, "returns");

    creditmc::VarEs out;
    {
        py::gil_scoped_release release;
        out = creditmc::historical_var_es(r, window, alpha);
    }
    return py::make_tuple(to_array(out.var), to_array(out.es));
}

py::tuple normal_var_es(const DoubleArray& returns, std::size_t window, double alpha) {
    const std::vector<double> r = to_vector(returns, "returns");

    creditmc::VarEs out;
    {
        py::gil_scoped_release release;
        out = creditmc::normal_var_es(r, window, alpha);
    }
    return py::make_tuple(to_array(out.var), to_array(out.es));
}

py::tuple ewma_var_es(const DoubleArray& returns, std::size_t window, double alpha, double lam) {
    const std::vector<double> r = to_vector(returns, "returns");

    creditmc::VarEs out;
    {
        py::gil_scoped_release release;
        out = creditmc::ewma_var_es(r, window, alpha, lam);
    }
    return py::make_tuple(to_array(out.var), to_array(out.es));
}

// Shared plumbing for the batch functions: copy the input arrays, release the GIL while the
// C++ runs, then turn each result into a (var, es) tuple of NumPy arrays.
template <typename Fn>
py::list run_batch_binding(const std::vector<DoubleArray>& series, Fn fn) {
    std::vector<std::vector<double>> data;
    data.reserve(series.size());
    for (const auto& s : series) {
        data.push_back(to_vector(s, "series"));
    }
    creditmc::SeriesList spans;
    spans.reserve(data.size());
    for (const auto& d : data) {
        spans.emplace_back(d);
    }

    std::vector<creditmc::VarEs> out;
    {
        py::gil_scoped_release release;
        out = fn(spans);
    }

    py::list result;
    for (const auto& r : out) {
        result.append(py::make_tuple(to_array(r.var), to_array(r.es)));
    }
    return result;
}

py::list historical_var_es_batch(const std::vector<DoubleArray>& series, std::size_t window,
                                 double alpha, unsigned n_threads) {
    return run_batch_binding(series, [&](const creditmc::SeriesList& s) {
        return creditmc::historical_var_es_batch(s, window, alpha, n_threads);
    });
}

py::list normal_var_es_batch(const std::vector<DoubleArray>& series, std::size_t window,
                             double alpha, unsigned n_threads) {
    return run_batch_binding(series, [&](const creditmc::SeriesList& s) {
        return creditmc::normal_var_es_batch(s, window, alpha, n_threads);
    });
}

py::list ewma_var_es_batch(const std::vector<DoubleArray>& series, std::size_t window,
                           double alpha, double lam, unsigned n_threads) {
    return run_batch_binding(series, [&](const creditmc::SeriesList& s) {
        return creditmc::ewma_var_es_batch(s, window, alpha, lam, n_threads);
    });
}

py::array_t<int> hits(const DoubleArray& returns, const DoubleArray& var) {
    const std::vector<int> h = creditmc::hits(to_vector(returns, "returns"), to_vector(var, "var"));
    py::array_t<int> result(static_cast<py::ssize_t>(h.size()));
    if (!h.empty()) {
        std::memcpy(result.mutable_data(), h.data(), h.size() * sizeof(int));
    }
    return result;
}

py::tuple to_tuple(const creditmc::TestResult& r) {
    return py::make_tuple(r.statistic, r.p_value);
}

py::tuple kupiec(const IntArray& h, double alpha) {
    return to_tuple(creditmc::kupiec(to_int_vector(h, "hits"), alpha));
}

py::tuple christoffersen_independence(const IntArray& h) {
    return to_tuple(creditmc::christoffersen_independence(to_int_vector(h, "hits")));
}

py::tuple conditional_coverage(const IntArray& h, double alpha) {
    return to_tuple(creditmc::conditional_coverage(to_int_vector(h, "hits"), alpha));
}

double es_ratio(const DoubleArray& returns, const DoubleArray& var, const DoubleArray& es) {
    return creditmc::es_ratio(to_vector(returns, "returns"), to_vector(var, "var"), to_vector(es, "es"));
}

std::string traffic_light(const IntArray& h, std::size_t window) {
    return creditmc::traffic_light(to_int_vector(h, "hits"), window);
}

double fz0(const DoubleArray& returns, const DoubleArray& var, const DoubleArray& es, double alpha) {
    return creditmc::fz0(to_vector(returns, "returns"), to_vector(var, "var"), to_vector(es, "es"), alpha);
}

}  // namespace

PYBIND11_MODULE(creditmc_core, m) {
    m.doc() = "Credit loss Monte Carlo engine and market-risk VaR/ES tools";

    m.def("simulate", &simulate,
          py::arg("pd"), py::arg("ead"), py::arg("lgd"), py::arg("rho"),
          py::arg("n_scenarios"), py::arg("seed") = 42, py::arg("n_threads") = 0);

    m.def("irb_asymptotic_loss", &creditmc::irb_asymptotic_loss,
          py::arg("pd"), py::arg("lgd"), py::arg("rho"), py::arg("confidence") = 0.999);

    m.def("norm_inv_cdf", &creditmc::norm_inv_cdf, py::arg("p"));
    m.def("norm_cdf", &creditmc::norm_cdf, py::arg("x"));

    m.def("historical_var_es", &historical_var_es,
          py::arg("returns"), py::arg("window") = 250, py::arg("alpha") = 0.05,
          "Rolling historical VaR and ES (positive losses). Returns (var, es), aligned with returns[window:].");

    m.def("normal_var_es", &normal_var_es,
          py::arg("returns"), py::arg("window") = 250, py::arg("alpha") = 0.05,
          "Rolling normal VaR and ES from the trailing sample mean and standard deviation.");

    m.def("ewma_var_es", &ewma_var_es,
          py::arg("returns"), py::arg("window") = 250, py::arg("alpha") = 0.05, py::arg("lam") = 0.94,
          "EWMA (zero-mean normal) VaR and ES. Output starts at index `window`.");

    m.def("historical_var_es_batch", &historical_var_es_batch,
          py::arg("series"), py::arg("window") = 250, py::arg("alpha") = 0.05, py::arg("n_threads") = 0,
          "Rolling historical VaR/ES for many series at once, in parallel across series. "
          "Returns a list of (var, es) tuples.");
    m.def("normal_var_es_batch", &normal_var_es_batch,
          py::arg("series"), py::arg("window") = 250, py::arg("alpha") = 0.05, py::arg("n_threads") = 0,
          "Rolling normal VaR/ES for many series at once. Returns a list of (var, es) tuples.");
    m.def("ewma_var_es_batch", &ewma_var_es_batch,
          py::arg("series"), py::arg("window") = 250, py::arg("alpha") = 0.05, py::arg("lam") = 0.94,
          py::arg("n_threads") = 0,
          "EWMA VaR/ES for many series at once. Returns a list of (var, es) tuples.");

    m.def("hits", &hits, py::arg("returns"), py::arg("var"),
          "Exceedance indicators: 1 where return < -VaR.");
    m.def("kupiec", &kupiec, py::arg("hits"), py::arg("alpha"),
          "Kupiec proportion-of-failures test. Returns (LR statistic, p-value).");
    m.def("christoffersen_independence", &christoffersen_independence, py::arg("hits"),
          "Christoffersen independence test. Returns (LR statistic, p-value).");
    m.def("conditional_coverage", &conditional_coverage, py::arg("hits"), py::arg("alpha"),
          "Conditional coverage test (Kupiec + independence). Returns (LR statistic, p-value).");
    m.def("es_ratio", &es_ratio, py::arg("returns"), py::arg("var"), py::arg("es"),
          "Mean loss/ES ratio on exceedance days; NaN if fewer than 5 exceedances.");
    m.def("traffic_light", &traffic_light, py::arg("hits"), py::arg("window") = 250,
          "Basel traffic-light zone ('green', 'yellow' or 'red'); meaningful for 99% VaR only.");
    m.def("fz0", &fz0, py::arg("returns"), py::arg("var"), py::arg("es"), py::arg("alpha"),
          "Mean Fissler-Ziegel FZ0 score of VaR/ES forecasts (lower is better).");
}