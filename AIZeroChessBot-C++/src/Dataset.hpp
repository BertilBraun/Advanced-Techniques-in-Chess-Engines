#pragma once

#include "common.hpp"

#include <future>

using DataSample = std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>;

class Dataset {
public:
    Dataset(const std::filesystem::path &savePath, torch::Device device,
            size_t memoriesToPreload = 10)
        : m_savePath(savePath), m_device(device), m_memoriesToPreload(memoriesToPreload) {

        if (!std::filesystem::exists(m_savePath / MEMORY_DIR_NAME)) {
            std::filesystem::create_directories(m_savePath / MEMORY_DIR_NAME);
        }

        restart();
    }

    void restart() {
        std::cout << "Loading memories from " << m_savePath / MEMORY_DIR_NAME << std::endl;

        auto newMemoryPaths = getMemoryPaths();

        for (const auto &memoryPath : newMemoryPaths) {
            if (std::find(m_memoryPathsFIFO.begin(), m_memoryPathsFIFO.end(), memoryPath) ==
                m_memoryPathsFIFO.end()) {
                m_memoryPathsFIFO.push_back(memoryPath);
            }
        }

        m_memoryPaths = m_memoryPathsFIFO; // copy the memory paths to the main list

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(m_memoryPaths.begin(), m_memoryPaths.end(), g);

        assert(m_memoriesToPreload > 0);

        for (m_currentMemoryIndex = 0; m_currentMemoryIndex < m_memoriesToPreload;
             ++m_currentMemoryIndex) {
            queueNextMemory();
        }

        std::cout << "Loaded " << newMemoryPaths.size() << " new memory batches "
                  << m_memoryPaths.size() << " in total." << std::endl;
    }

    bool hasNext() const { return !m_memoryFutures.empty(); }

    DataSample next() {
        if (m_memoryFutures.empty()) {
            throw std::runtime_error("No more memories to load");
        }

        auto result = m_memoryFutures.front().get();
        m_memoryFutures.pop_front();

        queueNextMemory();

        return result;
    }

    size_t size() const { return m_memoryPaths.size(); }

    void deleteOldMemories(int retentionFactor) {
        // delete the oldest retentionFactor% of the memories
        size_t numMemoriesToDelete = retentionFactor * m_memoryPaths.size() / 100;
        for (size_t i = 0; i < numMemoriesToDelete; ++i) {
            std::filesystem::remove_all(m_memoryPathsFIFO[i]);
        }

        // delete the oldest retentionFactor% of the memories
        m_memoryPathsFIFO.erase(m_memoryPathsFIFO.begin(),
                                m_memoryPathsFIFO.begin() + numMemoriesToDelete);
    }

private:
    std::deque<std::future<DataSample>> m_memoryFutures;
    std::vector<std::filesystem::path> m_memoryPathsFIFO;
    std::vector<std::filesystem::path> m_memoryPaths;
    std::filesystem::path m_savePath;
    torch::Device m_device;
    size_t m_currentMemoryIndex = 0;
    size_t m_memoriesToPreload;

    void queueNextMemory() {
        if (m_currentMemoryIndex < m_memoryPaths.size()) {
            auto memoryPath = m_memoryPaths[m_currentMemoryIndex++];

            m_memoryFutures.push_back(
                std::async(std::launch::async, &Dataset::loadMemory, this, memoryPath));
        }
    }

    std::vector<std::filesystem::path> getMemoryPaths() const {
        std::vector<std::filesystem::path> memoryPaths;
        for (const auto &entry :
             std::filesystem::directory_iterator(m_savePath / MEMORY_DIR_NAME)) {
            if (entry.is_directory() && isMemoryValid(entry.path())) {
                memoryPaths.push_back(entry.path());
            }
        }
        return memoryPaths;
    }

    DataSample loadMemory(const std::filesystem::path &memoryPath) const {
        torch::Tensor states, policyTargets, valueTargets;

        torch::load(states, (memoryPath / "states.pt").string(), m_device);
        torch::load(policyTargets, (memoryPath / "policyTargets.pt").string(), m_device);
        torch::load(valueTargets, (memoryPath / "valueTargets.pt").string(), m_device);

        return std::make_tuple(states, policyTargets, valueTargets);
    }

    bool isMemoryValid(const std::filesystem::path &memoryPath) const {
        return std::filesystem::exists(memoryPath / "states.pt") &&
               std::filesystem::exists(memoryPath / "policyTargets.pt") &&
               std::filesystem::exists(memoryPath / "valueTargets.pt");
    }
};
