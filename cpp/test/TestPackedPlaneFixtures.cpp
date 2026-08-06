#include "BoardEncoding.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

std::vector<std::int8_t> parseHexPayload(const std::string &hex) {
    if (hex.size() % 2 != 0) {
        throw std::runtime_error("Packed-plane fixture hex must have even length.");
    }
    std::vector<std::int8_t> payload(hex.size() / 2);
    for (std::size_t index = 0; index < payload.size(); ++index) {
        const std::string byteText = hex.substr(index * 2, 2);
        payload[index] = static_cast<std::int8_t>(std::stoi(byteText, nullptr, 16));
    }
    return payload;
}

std::string toHex(const std::vector<std::int8_t> &payload) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string hex;
    hex.reserve(payload.size() * 2);
    for (const std::int8_t byte : payload) {
        const auto value = static_cast<std::uint8_t>(byte);
        hex.push_back(digits[value >> 4]);
        hex.push_back(digits[value & 0x0F]);
    }
    return hex;
}

} // namespace

int main() {
    try {
        Bitboards::init();
        Position::init();

        const std::string fixturePath =
            std::string(SOURCE_ROOT) + "/../py/test/fixtures/chess_packed_plane_fixtures.json";
        std::ifstream fixtureStream(fixturePath);
        if (!fixtureStream) {
            std::cerr << "Failed to open fixture file: " << fixturePath << '\n';
            return 1;
        }
        const nlohmann::json fixtures = nlohmann::json::parse(fixtureStream);
        for (const nlohmann::json &fixture : fixtures) {
            const Board board(fixture.at("fen").get<std::string>());
            const CompressedEncodedBoard encoded = encodeBoard(&board);
            std::vector<std::int8_t> actualPayload(ChessPackedPlaneLayout::payload_bytes);
            writePackedPlaneEncoding(encoded, actualPayload.data());
            const std::vector<std::int8_t> expectedPayload =
                parseHexPayload(fixture.at("packed_hex").get<std::string>());
            if (actualPayload != expectedPayload) {
                std::cerr << fixture.at("fen").get<std::string>() << '\n';
                std::cerr << "expected: " << fixture.at("packed_hex").get<std::string>() << '\n';
                std::cerr << "actual:   " << toHex(actualPayload) << '\n';
                return 1;
            }
        }

        return 0;
    } catch (const std::exception &exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
