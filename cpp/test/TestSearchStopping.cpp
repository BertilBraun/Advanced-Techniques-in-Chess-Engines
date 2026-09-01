#include "TestRunner.hpp"
#include "search/SearchExecutor.hpp"
#include "search/SearchTypes.hpp"

#include <cmath>
#include <filesystem>
#include <memory>
#include <string>

namespace {

void require(const bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::filesystem::path saveModule(torch::jit::script::Module &model, const std::string &name) {
    const auto uniqueSuffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("search-stopping-test-" + name + "-" + std::to_string(uniqueSuffix) + ".jit.pt");
    model.save(path.string());
    return path;
}

std::filesystem::path constantPredictor(const std::string &name, const float uncertainty) {
    torch::jit::script::Module model("stop_predictor_constant");
    model.register_buffer("uncertainty", torch::tensor({uncertainty}));
    model.define(R"JIT(
        def forward(self, features):
            return self.uncertainty.unsqueeze(0).repeat((features.size(0), 1))
    )JIT");
    return saveModule(model, name);
}

// Reads one feature so probe evaluation and threshold comparisons exercise the input path.
std::filesystem::path movementEchoPredictor(const std::string &name) {
    torch::jit::script::Module model("stop_predictor_echo");
    model.define(R"JIT(
        def forward(self, features):
            return torch.clamp(features[:, 4:5], 0.0, 1.0)
    )JIT");
    return saveModule(model, name);
}

std::filesystem::path invalidShapePredictor(const std::string &name) {
    torch::jit::script::Module model("stop_predictor_invalid");
    model.define(R"JIT(
        def forward(self, features):
            return torch.zeros((features.size(0), 2))
    )JIT");
    return saveModule(model, name);
}

std::filesystem::path outOfRangePredictor(const std::string &name) {
    torch::jit::script::Module model("stop_predictor_range");
    model.define(R"JIT(
        def forward(self, features):
            return torch.full((features.size(0), 1), 1.5)
    )JIT");
    return saveModule(model, name);
}

StopPredictorFeatures featuresWithMovement(const double movement) {
    StopPredictorFeatures features{};
    features[4] = movement;
    return features;
}

} // namespace

int runSearchStoppingTests() {
    const std::filesystem::path certainPath = constantPredictor("certain", 0.05F);
    const std::filesystem::path echoPath = movementEchoPredictor("echo");
    const std::filesystem::path invalidPath = invalidShapePredictor("invalid");
    const std::filesystem::path rangePath = outOfRangePredictor("range");
    const auto cleanup = [&]() {
        std::filesystem::remove(certainPath);
        std::filesystem::remove(echoPath);
        std::filesystem::remove(invalidPath);
        std::filesystem::remove(rangePath);
    };
    try {
        {
            // stopPolicyKl mirrors src/search_stopping/targets.py::policy_kl exactly.
            const std::vector<double> reference = {0.7, 0.2, 0.1};
            const std::vector<double> approximate = {0.5, 0.4, 0.1};
            const double expected =
                0.7 * std::log(0.7 / 0.5) + 0.2 * std::log(0.2 / 0.4) + 0.1 * std::log(0.1 / 0.1);
            require(std::abs(stopPolicyKl(reference, approximate) - expected) < 1e-12,
                    "stopPolicyKl does not match the Python label math");
            const std::vector<double> zeroMass = {0.0, 1.0, 0.0};
            require(stopPolicyKl(zeroMass, reference) == 1.0 * std::log(1.0 / 0.2),
                    "stopPolicyKl did not skip zero-mass reference terms");
            require(stopPolicyKl(reference, zeroMass) > 0.0 &&
                        std::isfinite(stopPolicyKl(reference, zeroMass)),
                    "stopPolicyKl did not floor zero-mass approximations");
        }
        {
            const auto predictor = std::make_shared<SearchStopPredictor>(certainPath.string());
            const SearchStopPolicy policy({0.5, 1.0}, {0.1, 0.5}, 0.02, 2.0, predictor, true);
            // Guard precedence: movement above the epsilon blocks the predictor entirely.
            const StopCheckpointEvaluation blocked =
                evaluateStopRule(policy, 0, featuresWithMovement(0.5));
            require(blocked.guard_movement == 0.5 && !blocked.guard_passed &&
                        !blocked.predictor_evaluated && !blocked.would_stop,
                    "the movement guard did not block a visibly moving distribution");
            // Guard passed: u = 0.05 stops under threshold 0.1 but not under 0.05-tight rules.
            const StopCheckpointEvaluation stopped =
                evaluateStopRule(policy, 0, featuresWithMovement(0.001));
            require(stopped.guard_passed && stopped.predictor_evaluated &&
                        std::abs(stopped.uncertainty - 0.05) < 1e-6 && stopped.would_stop,
                    "a certain checkpoint under the guard did not stop");
            const StopCheckpointEvaluation attenuated = evaluateStopRule(
                SearchStopPolicy({0.5, 1.0}, {0.0, 0.5}, 0.02, 2.0, predictor, true), 0,
                featuresWithMovement(0.001));
            require(!attenuated.would_stop,
                    "an attenuated checkpoint (threshold zero) stopped a search");
            try {
                static_cast<void>(evaluateStopRule(policy, 2, featuresWithMovement(0.0)));
                throw std::runtime_error("an out-of-range checkpoint index validated");
            } catch (const std::invalid_argument &) {
            }
        }
        {
            // The uncertainty threshold comparison is strict: u < threshold stops.
            const auto predictor = std::make_shared<SearchStopPredictor>(echoPath.string());
            const SearchStopPolicy policy({0.5}, {0.25}, 1.0, 2.0, predictor, true);
            require(evaluateStopRule(policy, 0, featuresWithMovement(0.2)).would_stop,
                    "uncertainty below the threshold did not stop");
            require(!evaluateStopRule(policy, 0, featuresWithMovement(0.25)).would_stop,
                    "uncertainty at the threshold stopped");
        }
        {
            // Probe validation at load: wrong shape, out-of-range output, missing file.
            try {
                static_cast<void>(SearchStopPredictor(invalidPath.string()));
                throw std::runtime_error("a wrong-shape predictor unexpectedly loaded");
            } catch (const std::invalid_argument &) {
            }
            try {
                static_cast<void>(SearchStopPredictor(rangePath.string()));
                throw std::runtime_error("an out-of-range predictor unexpectedly loaded");
            } catch (const std::invalid_argument &) {
            }
            try {
                static_cast<void>(SearchStopPredictor("/nonexistent/stop-predictor.jit.pt"));
                throw std::runtime_error("a missing predictor unexpectedly loaded");
            } catch (const std::invalid_argument &) {
            }
        }
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
    return 0;
}
