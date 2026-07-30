#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace az::v2::search {

class SeededRandom {
public:
    explicit SeededRandom(std::uint64_t seed) : engine_(seed) {}

    [[nodiscard]] std::size_t sample_discrete(const std::vector<double> &weights) {
        double total = 0.0;
        for (const double weight : weights) {
            if (!std::isfinite(weight) || weight < 0.0) {
                throw std::invalid_argument("sampling weights must be finite and non-negative");
            }
            total += weight;
        }
        if (!(total > 0.0) || !std::isfinite(total)) {
            throw std::invalid_argument("sampling weights must have positive finite mass");
        }
        const double draw = uniform_(engine_) * total;
        double cumulative = 0.0;
        for (std::size_t index = 0; index < weights.size(); ++index) {
            cumulative += weights[index];
            if (draw < cumulative) {
                return index;
            }
        }
        return weights.size() - 1;
    }

    [[nodiscard]] std::vector<double> dirichlet(std::size_t size, double alpha) {
        if (size == 0 || !std::isfinite(alpha) || !(alpha > 0.0)) {
            throw std::invalid_argument(
                "Dirichlet requires a non-empty shape and finite positive alpha");
        }
        std::gamma_distribution<double> gamma(alpha, 1.0);
        std::vector<double> samples(size);
        double total = 0.0;
        for (double &sample : samples) {
            sample = gamma(engine_);
            total += sample;
        }
        if (!(total > 0.0) || !std::isfinite(total)) {
            throw std::runtime_error("Dirichlet sampling produced invalid mass");
        }
        for (double &sample : samples) {
            sample /= total;
        }
        return samples;
    }

private:
    std::mt19937_64 engine_;
    std::uniform_real_distribution<double> uniform_{0.0, 1.0};
};

} // namespace az::v2::search
