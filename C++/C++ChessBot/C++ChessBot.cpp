
#include <iostream>
#include <SFML/Graphics.hpp>

#include "thc-Chess/thc.h"

#include "Defines.h"

//#pragma region THC_HOOKS

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

//#pragma endregion

thc::Square toSquare(int x, int y) {
    return (thc::Square)(x + y * 8);
}

std::vector<sf::RectangleShape> tiles;
std::map<char, sf::Texture> imageDict;

void InitDrawables() {
    sf::Color darkColor(209, 139, 71);
    sf::Color lightColor(255, 206, 158);

    for (int x = 0; x < 8; x++)
        for (int y = 0; y < 8; y++) {
            sf::RectangleShape tile;
            if ((y + x * 8 + x) % 2 == 0)
                tile.setFillColor(darkColor);
            else
                tile.setFillColor(lightColor);
            tile.setPosition(x * 80, y * 80);
            tile.setSize({ 80, 80 });
            tiles.push_back(tile);
        }

    imageDict['P'].loadFromFile("Res/WPawn.png");
    imageDict['R'].loadFromFile("Res/WRook.png");
    imageDict['N'].loadFromFile("Res/WKnight.png");
    imageDict['B'].loadFromFile("Res/WBishop.png");
    imageDict['Q'].loadFromFile("Res/WQueen.png");
    imageDict['K'].loadFromFile("Res/WKing.png");

    imageDict['p'].loadFromFile("Res/BPawn.png");
    imageDict['r'].loadFromFile("Res/BRook.png");
    imageDict['n'].loadFromFile("Res/BKnight.png");
    imageDict['b'].loadFromFile("Res/BBishop.png");
    imageDict['q'].loadFromFile("Res/BQueen.png");
    imageDict['k'].loadFromFile("Res/BKing.png");
}

void PrintBoard(sf::RenderWindow& window) {

    for (auto tile : tiles)
        window.draw(tile);
}

void PrintGame(thc::ChessEngine& engine, sf::RenderWindow& window) {

    window.clear();

    PrintBoard(window);

    for (int x = 0; x < 8; x++)
        for (int y = 0; y < 8; y++) {
            char piece = engine.squares[x + y * 8];
            if (piece != ' ') {
                auto image = imageDict[piece];
                sf::Sprite sprite(image);
                sprite.setPosition(x * 80, y * 80);
                sprite.setScale(0.08, 0.08);
                window.draw(sprite);
            }
        }

    window.display();
}

bool isMate(thc::ChessEngine& engine, char color = ' ') {

    bool mate = false;

    switch (color)
    {
    case ' ':
        if (engine.AttackedPiece((thc::Square)engine.bking_square))
        {
            mate = true;

            thc::MOVELIST list;
            engine.GenLegalMoveList(&list);
            for (int i = 0; mate && i < list.count; i++)
            {
                engine.PushMove(list.moves[i]);
                if (!engine.AttackedPiece((thc::Square)engine.bking_square))
                    mate = false;
                engine.PopMove(list.moves[i]);
            }
        }
    case 'w':
        if (!mate && engine.AttackedPiece((thc::Square)engine.wking_square))
        {
            mate = true;

            thc::MOVELIST list;
            engine.GenLegalMoveList(&list);
            for (int i = 0; mate && i < list.count; i++)
            {
                engine.PushMove(list.moves[i]);
                if (!engine.AttackedPiece((thc::Square)engine.wking_square))
                    mate = false;
                engine.PopMove(list.moves[i]);
            }
        }
        break;
    case 'b':
        if (engine.AttackedPiece((thc::Square)engine.bking_square))
        {
            mate = true;

            thc::MOVELIST list;
            engine.GenLegalMoveList(&list);
            for (int i = 0; mate && i < list.count; i++)
            {
                engine.PushMove(list.moves[i]);
                if (!engine.AttackedPiece((thc::Square)engine.bking_square))
                    mate = false;
                engine.PopMove(list.moves[i]);
            }
        }
        break;
    }

    return mate;
}

bool PlayerMove(thc::ChessEngine& engine, sf::RenderWindow& window, int fx, int fy, int tx, int ty) {
    thc::Move move;
    move.src = toSquare(fx, fy);
    move.dst = toSquare(tx, ty);

    std::vector<thc::Move> moves;
    engine.GenLegalMoveList(moves);

    for (auto m : moves)
        if (m == move && ! isMate(engine, 'w')) {
            engine.PlayMove(m);
            PrintGame(engine, window);

            return true;
        }

    std::cout << "Entered Move was not valid, try again!" << std::endl;

    PrintGame(engine, window);

    return false;
}

int minimax(thc::ChessEngine& engine, int depth, int alpha, int beta, bool maximizing);

int totalEvaluatedMoves = 0;

void BotMove(thc::ChessEngine& engine, sf::RenderWindow& window) {

    if (isMate(engine, 'b'))
        return;

    bool only_move;
    int score;
    thc::Move move;

    // TODO Analyze
    if (engine.CalculateNextMove(only_move, score, move, 4, 6))
    {
        engine.PlayMove(move);
        PrintGame(engine, window);
        return;
    }
    else
        std::cout << "Not Found" << std::endl;

    totalEvaluatedMoves = 0;

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

    std::cout << "Evaluated: " << totalEvaluatedMoves << " seperate Moves this Turn!" << std::endl;
    PrintGame(engine, window);
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

    totalEvaluatedMoves++;
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

            maxVal = std::max(maxVal, val);
            alpha = std::max(alpha, val);
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

            minVal = std::min(minVal, val);
            beta = std::min(beta, val);
            if (beta <= alpha)
                break;
        }
        return minVal;
    }
}

int main()
{

    thc::ChessEngine engine;
    sf::RenderWindow window(sf::VideoMode(80 * 8, 80 * 8), "Chess Bot!");

    InitDrawables();
    PrintGame(engine, window);

    int fx = 0, fy = 0;

    while (true) {

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed) {
                window.close();
                return 0;
            }
            else if (event.type == sf::Event::MouseButtonPressed &&
                event.mouseButton.button == sf::Mouse::Left)
            {
                fx = event.mouseButton.x / 80;
                fy = event.mouseButton.y / 80;
            }
            else if (event.type == sf::Event::MouseButtonReleased &&
                event.mouseButton.button == sf::Mouse::Left) {

                int tx = event.mouseButton.x / 80;
                int ty = event.mouseButton.y / 80;

                if (PlayerMove(engine, window, fx, fy, tx, ty))
                    BotMove(engine, window);
            }
        }

        if (isMate(engine))
            break;
    }

    if (engine.AttackedPiece((thc::Square)engine.wking_square))
        std::cout << "Black Won!" << std::endl;
    else
        std::cout << "White Won!" << std::endl;

    Sleep(2000);
    return 0;
}
