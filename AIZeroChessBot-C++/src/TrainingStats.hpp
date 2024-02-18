#pragma once

#include "common.hpp"

class TrainingStats {
public:
    float policyLoss = 0.0f;
    float valueLoss = 0.0f;
    float totalLoss = 0.0f;
    int numBatches = 0;

    void update(float policyLoss, float valueLoss, float totalLoss) {
        this->policyLoss += policyLoss;
        this->valueLoss += valueLoss;
        this->totalLoss += totalLoss;
        ++numBatches;
    }

    // Overload the + operator to add two TrainingStats objects
    TrainingStats operator+(const TrainingStats &other) const {
        return TrainingStats{policyLoss + other.policyLoss, valueLoss + other.valueLoss,
                             totalLoss + other.totalLoss, numBatches + other.numBatches};
    }
    TrainingStats &operator+=(const TrainingStats &other) {
        *this = *this + other;
        return *this;
    }

    std::string toString() const {
        std::ostringstream stream;
        stream << "Policy Loss: " << policyLoss / numBatches
               << ", Value Loss: " << valueLoss / numBatches
               << ", Total Loss: " << totalLoss / numBatches;
        return stream.str();
    }
};

class LearningStats {
public:
    int totalNumGames = 0;
    int totalIterations = 0;
    std::vector<TrainingStats> trainingStats;

    void update(int numGames, const TrainingStats &stats) {
        totalNumGames += numGames;
        trainingStats.push_back(stats);
        ++totalIterations;
    }

    std::string toString() const {
        std::ostringstream stream;
        stream << "Total Games: " << totalNumGames << "\n";
        stream << "Total Iterations: " << totalIterations << "\n";
        for (size_t i = 0; i < trainingStats.size(); ++i) {
            stream << "Iteration " << i + 1 << ": " << trainingStats[i].toString() << "\n";
        }
        return stream.str();
    }
};
