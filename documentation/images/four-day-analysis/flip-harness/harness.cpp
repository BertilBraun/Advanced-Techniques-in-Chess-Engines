// Standalone harness around the REAL C++ encoders in /tmp/az/head/cpp.
// stdin: one FEN per line.  For each FEN prints:
//   FEN <fen>
//   PLANES <29*64 chars of digits/signed ints separated by ','>  (tensor index = plane*64 + square, square = rank*8+file, a1=0)
//   MOVE <uci> <actionId> <decodedUci> <mirrorActionId>
//   END
#include "games/chess/encoding/ChessEncoding.hpp"
#include "games/chess/implementation/ChessBoard.hpp"
#include "bitboard.h"
#include "position.h"

#include <iostream>
#include <string>
#include <vector>

int main() {
    Stockfish::Bitboards::init();
    Stockfish::Position::init();
    std::string fen;
    constexpr int values = ChessRepresentationDimensions::channel_count * 64;
    std::vector<std::int8_t> buf(values);
    while (std::getline(std::cin, fen)) {
        if (fen.empty()) continue;
        Board board(fen);
        ChessEncoding::encodeInputInto(board, buf.data());
        std::cout << "FEN " << fen << "\n";
        std::cout << "PLANES";
        for (int i = 0; i < values; ++i) std::cout << (i ? "," : " ") << int(buf[i]);
        std::cout << "\n";
        {
            const CompressedEncodedBoard enc = encodeBoard(board);
            std::string payload(CompressedEncodedBoard::packed_bytes, '\0');
            enc.writePackedInto(std::span(reinterpret_cast<std::int8_t *>(payload.data()), payload.size()));
            std::cout << "PACKED ";
            static const char *hex = "0123456789abcdef";
            for (unsigned char c : payload) std::cout << hex[c >> 4] << hex[c & 15];
            std::cout << "\n";
        }
        for (const Stockfish::Move m : board.validMoves()) {
            const ChessAction a(m);
            const int id = ChessEncoding::actionId(a, board);
            std::string decoded;
            try {
                decoded = ChessEncoding::decodeAction(id, board).toUci();
            } catch (const std::exception &e) {
                decoded = std::string("ERR:") + e.what();
            }
            std::cout << "MOVE " << a.toUci() << " " << id << " " << decoded << " " << int(m.from_sq()) << " " << int(m.to_sq()) << " "
                      << ChessEncoding::mirrorActionId(id) << "\n";
        }
        std::cout << "END\n";
    }
    return 0;
}
