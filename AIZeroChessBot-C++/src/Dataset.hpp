#pragma once

#include "common.hpp"

#include <future>

using DataSample = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;

class DataSubset {
public:
    DataSubset(const std::vector<std::filesystem::path> &memoryPaths, torch::Device device,
               size_t memoryBatchSize, long long memoriesToPreload = 10)
        : m_memoryPaths(memoryPaths), m_device(device), m_memoriesToPreload(memoriesToPreload),
          m_memoryBatchSize(memoryBatchSize) {

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(m_memoryPaths.begin(), m_memoryPaths.end(), g);

        assert(m_memoriesToPreload > 0);

        for (m_currentMemoryIndex = 0; m_currentMemoryIndex < m_memoriesToPreload;
             ++m_currentMemoryIndex) {
            queueNextMemory();
        }
    }

    bool hasNext() const { return !m_memoryFutures.empty(); }

    DataSample next() {
        if (m_memoryFutures.empty()) {
            throw std::runtime_error("No more memories to load");
        }

        auto result = m_memoryFutures.front().get();
        m_memoryFutures.pop_front();

        while (std::get<0>(result).numel() == 0 || std::get<1>(result).numel() == 0 ||
               std::get<2>(result).numel() == 0) {
            log("Invalid memory detected. Skipping to the next one.");
            queueNextMemory();

            result = m_memoryFutures.front().get();
            m_memoryFutures.pop_front();
        }

        queueNextMemory();

        return result;
    }

private:
    std::deque<std::future<DataSample>> m_memoryFutures;
    std::vector<std::filesystem::path> m_memoryPaths;
    torch::Device m_device;
    size_t m_currentMemoryIndex = 0;
    size_t m_memoriesToPreload;
    long long m_memoryBatchSize;

    friend class Dataset;

    void queueNextMemory() {
        if (m_currentMemoryIndex < m_memoryPaths.size()) {
            auto memoryPath = m_memoryPaths[m_currentMemoryIndex++];

            m_memoryFutures.push_back(
                std::async(std::launch::async, &DataSubset::loadMemory, this, memoryPath));
        }
    }

    DataSample loadMemory(const std::filesystem::path &memoryPath) {
        torch::Tensor states, policyTargets, valueTargets;

        try {
            torch::load(states, (memoryPath / "states.pt").string());
            torch::load(policyTargets, (memoryPath / "policyTargets.pt").string());
            torch::load(valueTargets, (memoryPath / "valueTargets.pt").string());
        } catch (const c10::Error &e) {
            log("Error loading memory:", memoryPath);
            log("Error:", split(e.what(), '\n')[0]);

            // Delete the memory
            std::filesystem::remove_all(memoryPath);

            return std::make_tuple(torch::Tensor(), torch::Tensor(), torch::Tensor());
        }

        // if valueTargets is not of [batchSize, 1] shape -> error
        // if policyTargets is not of [batchSize, ACTION_SIZE] shape -> error
        // if states is not of [batchSize, 12, 8, 8] shape -> error
        if (states.sizes() != torch::IntArrayRef({m_memoryBatchSize, 12, 8, 8}) ||
            policyTargets.sizes() != torch::IntArrayRef({m_memoryBatchSize, ACTION_SIZE}) ||
            valueTargets.sizes() != torch::IntArrayRef({m_memoryBatchSize, 1})) {
            return std::make_tuple(torch::Tensor(), torch::Tensor(), torch::Tensor());
        }

        states = states.to(m_device);
        policyTargets = policyTargets.to(m_device);
        valueTargets = valueTargets.to(m_device);

        return std::make_tuple(states, policyTargets, valueTargets);
    }
};

class Dataset {
public:
    Dataset(size_t memoryBatchSize, size_t memoriesToPreload = 10)
        : m_memoriesToPreload(memoriesToPreload), m_memoryBatchSize((long long) memoryBatchSize) {

        if (!std::filesystem::exists(MEMORY_DIR)) {
            std::filesystem::create_directories(MEMORY_DIR);
        }
    }

    void load() {
        log("Loading memories from", MEMORY_DIR);

        auto newMemoryPaths = getMemoryPaths();

        size_t oldMemoryPathsSize = m_memoryPathsFIFO.size();

        std::set<std::filesystem::path> oldMemoryPaths(m_memoryPathsFIFO.begin(),
                                                       m_memoryPathsFIFO.end());

        // for all items in m_memoryPathsFIFO, if it is not in newMemoryPaths, delete it
        m_memoryPathsFIFO.erase(std::remove_if(m_memoryPathsFIFO.begin(), m_memoryPathsFIFO.end(),
                                               [&newMemoryPaths](const auto &memoryPath) {
                                                   return newMemoryPaths.find(memoryPath) ==
                                                          newMemoryPaths.end();
                                               }),
                                m_memoryPathsFIFO.end());

        // for all the new memory paths, if it is not in oldMemoryPaths, append it to
        // m_memoryPathsFIFO this will ensure that the new memory paths are deleted last
        for (const auto &memoryPath : newMemoryPaths) {
            if (oldMemoryPaths.find(memoryPath) == oldMemoryPaths.end()) {
                m_memoryPathsFIFO.push_back(memoryPath);
            }
        }

        log("Loaded", newMemoryPaths.size() - oldMemoryPathsSize, "new memory batches",
            m_memoryPathsFIFO.size(), "in total.");
    }

    size_t size() const { return m_memoryPathsFIFO.size(); }

    void deleteOldMemories(int retentionFactor) {
        // delete the oldest retentionFactor% of the memories
        float percentage = (1.0f - ((float) retentionFactor / 100.f));
        size_t numMemoriesToDelete = percentage * m_memoryPathsFIFO.size();
        for (size_t i = 0; i < numMemoriesToDelete; ++i) {
            std::filesystem::remove_all(m_memoryPathsFIFO[i]);
        }

        // delete the oldest retentionFactor% of the memories
        m_memoryPathsFIFO.erase(m_memoryPathsFIFO.begin(),
                                m_memoryPathsFIFO.begin() + numMemoriesToDelete);
    }

    std::vector<DataSubset> intoDataSubsets(const std::vector<torch::Device> &devices) const {
        std::vector<DataSubset> datasets;
        datasets.reserve(devices.size());

        size_t batchSizePerDevice = m_memoryBatchSize / devices.size();

        for (size_t i = 0; i < devices.size(); ++i) {

            std::vector<std::filesystem::path> memoryPaths;
            memoryPaths.reserve(batchSizePerDevice);

            for (size_t j = i * batchSizePerDevice;
                 j < (i + 1) * batchSizePerDevice && j < m_memoryPathsFIFO.size(); ++j) {
                memoryPaths.push_back(m_memoryPathsFIFO[j]);
            }

            datasets.emplace_back(memoryPaths, devices[i], batchSizePerDevice,
                                  (long long) m_memoriesToPreload);
        }

        return datasets;
    }

private:
    std::vector<std::filesystem::path> m_memoryPathsFIFO;
    size_t m_memoriesToPreload;
    long long m_memoryBatchSize;

    std::set<std::filesystem::path> getMemoryPaths() const {
        std::set<std::filesystem::path> memoryPaths;
        for (const auto &entry : std::filesystem::directory_iterator(MEMORY_DIR)) {
            if (entry.is_directory() && isMemoryValid(entry.path())) {
                memoryPaths.insert(entry.path());
            }
        }
        return memoryPaths;
    }

    bool isMemoryValid(const std::filesystem::path &memoryPath) const {
        return true; // Ignore for now, we will catch the error in loadMemory

        std::set<std::string> requiredFiles = {"states.pt", "policyTargets.pt", "valueTargets.pt"};
        std::set<std::string> foundFiles;

        for (const auto &entry : std::filesystem::directory_iterator(memoryPath)) {
            if (entry.is_regular_file()) {
                foundFiles.insert(entry.path().filename().string());
            }
        }

        return std::includes(foundFiles.begin(), foundFiles.end(), requiredFiles.begin(),
                             requiredFiles.end());
    }
};
