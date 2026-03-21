#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <fstream>

namespace pgn {

// Represents a single parsed PGN game.
struct Game {
    std::unordered_map<std::string, std::string> headers;
    std::vector<std::string> moves; // Extracted SAN moves (clean)
    std::string result;             // e.g. "1-0", "0-1", "1/2-1/2", "*"
};

// Fast stream parser that yields Game objects lazily.
class PGNStream {
public:
    explicit PGNStream(const std::string& filepath);
    ~PGNStream();

    // Returns the next game in the stream, or std::nullopt if EOF.
    std::optional<Game> next();

private:
    std::ifstream file_;
};

} // namespace pgn