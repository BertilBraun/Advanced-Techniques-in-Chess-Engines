#include "common.hpp"

void selfPlayMain(int runId, const std::string &savePath, int numProcessors, int numGPUs);

typedef std::pair<float, std::vector<std::pair<int, int>>>
    PyInferenceResult; // (score, (move, visits));

std::vector<PyInferenceResult> boardInferenceMain(const std::string &modelPath,
                                                  const std::vector<std::string> &fens);

Move evalBoardIterate(const std::string &modelPath, const std::string &fen, bool networkOnly,
                      float maxTime);

TensorBoardLogger getTensorBoardLogger(int runId);