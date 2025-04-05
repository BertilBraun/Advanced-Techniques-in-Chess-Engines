#include "common.hpp"

void selfPlayMain(int runId, const std::string &savePath, int numProcessors, int numGPUs);

typedef std::pair<int, std::vector<std::pair<int, int>>> PyInferenceResult; // (score, visits);

std::vector<PyInferenceResult> boardInferenceMain(const std::string &modelPath,
                                                  const std::vector<std::string> &fens);

Move evalBoardIterate(const std::string &modelPath, const std::string &fen, bool networkOnly,
                      float maxTime);