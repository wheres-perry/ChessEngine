#include "pgn.hpp"
#include <cctype>
#include <iostream>

namespace pgn {

PGNStream::PGNStream(const std::string &filepath) : file_(filepath) {
  if (!file_.is_open()) {
    throw std::runtime_error("Failed to open PGN file: " + filepath);
  }
}

PGNStream::~PGNStream() {
  if (file_.is_open()) {
    file_.close();
  }
}

std::optional<Game> PGNStream::next() {
  if (!file_.is_open() || file_.eof()) {
    return std::nullopt;
  }

  Game game;
  std::string line;
  bool reading_moves = false;
  std::string move_text;

  // Phase 1: Read lines until the end of the game
  while (std::getline(file_, line)) {
    // Trim trailing \r (CRLF issues)
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }

    // Strip line comments
    size_t semi = line.find(';');
    if (semi != std::string::npos) {
      line = line.substr(0, semi);
    }

    if (line.empty()) {
      if (reading_moves && !move_text.empty()) {
        // An empty line while reading moves might denote the end of a game
        // in some loose PGN formats, but standard PGN expects the result
        // marker. We'll keep reading to be safe.
      }
      continue;
    }

    if (line[0] == '[') {
      if (reading_moves) {
        // Encountered new headers while reading moves (e.g., missing empty
        // line). We need to seek back so the next iteration processes this
        // line. But std::getline doesn't seek back cleanly without offsets. For
        // a highly robust parser we'd handle this, but standard PGN files
        // always terminate the move section with a result marker.
        break;
      }

      // Parse header: [Key "Value"]
      size_t space_pos = line.find(' ');
      if (space_pos != std::string::npos) {
        std::string key = line.substr(1, space_pos - 1);
        size_t quote1 = line.find('"', space_pos);
        size_t quote2 = line.rfind('"');
        if (quote1 != std::string::npos && quote2 != std::string::npos &&
            quote2 > quote1) {
          game.headers[key] = line.substr(quote1 + 1, quote2 - quote1 - 1);
        }
      }
    } else {
      reading_moves = true;
      move_text += line + " ";

      // Fast-check for game termination in this line to break out early.
      // A result marker ends the game.
      if (line.find("1-0") != std::string::npos ||
          line.find("0-1") != std::string::npos ||
          line.find("1/2-1/2") != std::string::npos ||
          line.find("*") != std::string::npos) {
        break;
      }
    }
  }

  if (!reading_moves && game.headers.empty()) {
    return std::nullopt; // Reached EOF completely
  }

  // Phase 2: Tokenize and clean the move text
  int rav_depth = 0;
  bool in_comment = false;
  std::string current_token;

  auto flush_token = [&]() {
    if (current_token.empty())
      return;

    // Identify if it's a result marker
    if (current_token == "1-0" || current_token == "0-1" ||
        current_token == "1/2-1/2" || current_token == "*") {
      game.result = current_token;
    }
    // Identify move numbers (contains '.')
    else if (current_token.find('.') != std::string::npos) {
      // It's a move number like "1.", "12...", skip it.
    }
    // Identify Numeric Annotation Glyphs (NAGs)
    else if (current_token[0] == '$') {
      // It's a NAG like "$1", "$3", skip it.
    } else {
      // It is a valid SAN string
      game.moves.push_back(current_token);
    }
    current_token.clear();
  };

  for (size_t i = 0; i < move_text.size(); ++i) {
    char c = move_text[i];

    if (in_comment) {
      if (c == '}')
        in_comment = false;
      continue;
    }

    if (c == '{') {
      flush_token();
      in_comment = true;
      continue;
    }

    if (c == '(') {
      flush_token();
      rav_depth++;
      continue;
    }

    if (c == ')') {
      flush_token();
      if (rav_depth > 0)
        rav_depth--;
      continue;
    }

    if (rav_depth > 0) {
      continue;
    }

    // We handle standard whitespace grouping
    if (std::isspace(static_cast<unsigned char>(c))) {
      flush_token();
    } else {
      current_token += c;
    }
  }

  flush_token(); // Flush the last token

  // If no explicit result was found but headers contain one, fallback to header
  if (game.result.empty()) {
    auto it = game.headers.find("Result");
    if (it != game.headers.end()) {
      game.result = it->second;
    }
  }

  return game;
}

} // namespace pgn