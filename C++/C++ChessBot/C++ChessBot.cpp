
#include <iostream>

#include "thc-Chess/thc.h"

#include "Defines.h"

// Just reproduce printf, a hook required by the thc library
int core_printf(const char* fmt, ...)
{
    int ret = 0;
    va_list args;
    va_start(args, fmt);
    char buf[1000];
    char* p = buf;
    vsnprintf(p, sizeof(buf) - 2 - (p - buf), fmt, args);
    fputs(buf, stdout);
    va_end(args);
    return ret;
}
// Stub out another hook required by the thc library
void ReportOnProgress(bool init, int multipv, std::vector<thc::Move>& pv, int score_cp, int depth)
{
}


thc::Square toSquare(int x, int y) {
    return (thc::Square)(x + y * 8);
}

bool PlayerMove(thc::ChessEngine& engine) {
    int fx, fy, tx, ty;
    std::cout << "Move from x: ";
    std::cin >> fx;
    std::cout << "and from y: ";
    std::cin >> fy;
    std::cout << "To x: ";
    std::cin >> tx;
    std::cout << "and y: ";
    std::cin >> ty;

    thc::Move move;
    move.src = toSquare(fx, fy);
    move.dst = toSquare(tx, ty);

    std::vector<thc::Move> moves;
    engine.GenLegalMoveList(moves);

    for (auto m : moves)
        if (m == move) {
            engine.PlayMove(move);
            std::cout << engine.ToDebugStr() << std::endl;

            return true;
        }

    std::cout << "Entered Move was not valid, try again!" << std::endl;

    std::cout << engine.ToDebugStr() << std::endl;

    return false;
}

int minimax(thc::ChessEngine& engine, int depth, int alpha, int beta, bool maximizing);

void BotMove(thc::ChessEngine& engine) {

    thc::Move bestMove;
    int maxVal = -9999999;

    std::vector<thc::Move> moves;
    engine.GenLegalMoveList(moves);

    for (auto m : moves) {

        engine.PushMove(m);
        int val = minimax(engine, MAX_DEPTH, -9999999, 9999999, true);
        engine.PopMove(m);

        if (val > maxVal) {
            maxVal = val;
            bestMove = m;
        }
    }

    engine.PlayMove(bestMove);

    std::cout << engine.ToDebugStr() << std::endl;
}

int evaluate(thc::ChessEngine& engine) {
    auto evaluatePiece = [=](char piece, int x, int y) {
        if (piece == ' ')
            return 0;

        char symbol = toupper(piece);
        auto& valueArray = pieceValueArrayDict.at(symbol);

        if (islower(piece)) {
            x = 7 - x;
            y = 7 - y;
        }

        int value = pieceValueDict.at(symbol) + valueArray[y][x];

        if (islower(piece))
            return value;
        else
            return -value;
    };

    int total = 0;

    for (int x = 0; x < 8; x++)
        for (int y = 0; y < 8; y++)
            total += evaluatePiece(engine.squares[x + y * 8], x, y);

    return total;
}

int minimax(thc::ChessEngine& engine, int depth, int alpha, int beta, bool maximizing) {
    thc::TERMINAL score_terminal = thc::TERMINAL::NOT_TERMINAL;
    engine.Evaluate(score_terminal);

    if (depth == 0 or score_terminal != thc::TERMINAL::NOT_TERMINAL)
        return evaluate(engine);

    if (maximizing) {
        int maxVal = -9999999;

        std::vector<thc::Move> moves;
        engine.GenLegalMoveList(moves);

        for (auto m : moves) {

            engine.PushMove(m);
            int val = minimax(engine, depth - 1, alpha, beta, false);
            engine.PopMove(m);

            maxVal = max(maxVal, val);
            alpha = max(alpha, val);
            if (beta <= alpha)
                break;
        }
        return maxVal;
    }
    else {
        int minVal = 9999999;

        std::vector<thc::Move> moves;
        engine.GenLegalMoveList(moves);

        for (auto m : moves) {

            engine.PushMove(m);
            int val = minimax(engine, depth - 1, alpha, beta, true);
            engine.PopMove(m);

            minVal = min(minVal, val);
            beta = min(beta, val);
            if (beta <= alpha)
                break;
        }
        return minVal;
    }
}


int main()
{
    thc::ChessEngine engine;
    thc::TERMINAL score_terminal = thc::TERMINAL::NOT_TERMINAL;

    while (score_terminal == thc::TERMINAL::NOT_TERMINAL) {

        if (!PlayerMove(engine))
            continue;

        BotMove(engine);

        engine.Evaluate(score_terminal);
    }

    if (score_terminal < 0)
        std::cout << "Black Won!" << std::endl;
    else
        std::cout << "White Won!" << std::endl;

    std::cin.get();
}
