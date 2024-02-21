#pragma once

#include "common.hpp"

#include <future>

class Dataset {
public:
    Dataset(const std::filesystem::path &savePath, torch::Device device,
            size_t memoriesToPreload = 10)
        : m_savePath(savePath), m_device(device) {

        if (!std::filesystem::exists(m_savePath / MEMORY_DIR_NAME)) {
            std::filesystem::create_directories(m_savePath / MEMORY_DIR_NAME);
        }

        m_memoryPaths = getMemoryPaths();

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(m_memoryPaths.begin(), m_memoryPaths.end(), g);

        assert(memoriesToPreload > 0);

        for (m_currentMemoryIndex = 0; m_currentMemoryIndex < memoriesToPreload;
             ++m_currentMemoryIndex) {
            queueNextMemory();
        }
    }

    bool hasNext() const { return !m_memoryFutures.empty(); }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> next() {
        if (m_memoryFutures.empty()) {
            throw std::runtime_error("No more memories to load");
        }

        auto result = m_memoryFutures.front().get();
        m_memoryFutures.pop_front();

        queueNextMemory();

        return result;
    }

    size_t size() const { return m_memoryPaths.size(); }

    void deleteOldMemories(int retentionFactor) const {
        for (const auto &path : m_memoryPaths) {
            if (std::filesystem::exists(path) && std::filesystem::is_directory(path) &&
                isMemoryValid(path)) {
                if (rand() % 100 > retentionFactor) {
                    // delete the entries in the folder as well as the folder itself
                    std::filesystem::remove_all(path);
                }
            }
        }
    }

private:
    std::deque<std::future<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>>>
        m_memoryFutures;
    std::vector<std::filesystem::path> m_memoryPaths;
    std::filesystem::path m_savePath;
    torch::Device m_device;
    size_t m_currentMemoryIndex = 0;

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

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    loadMemory(const std::filesystem::path &memoryPath) const {
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
