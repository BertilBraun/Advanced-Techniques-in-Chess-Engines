#pragma once

#include <array>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <type_traits>

template <std::size_t BoardSize> class BitBoard {
public:
    static_assert(BoardSize >= 1);
    static_assert(BoardSize <= 255);

    static constexpr std::size_t width = BoardSize;
    static constexpr std::size_t height = BoardSize;
    static constexpr std::size_t bit_count = BoardSize * BoardSize;
    static constexpr std::size_t word_bits = 64;
    static constexpr std::size_t word_count = (bit_count + word_bits - 1) / word_bits;

    using Word = std::uint64_t;
    using Storage = std::array<Word, word_count>;

    struct Point {
        std::uint8_t x;
        std::uint8_t y;

        friend constexpr bool operator==(Point, Point) = default;
    };

    constexpr BitBoard() noexcept = default;

    explicit constexpr BitBoard(Storage words) noexcept : words_(words) { clear_unused_bits(); }

    [[nodiscard]]
    static constexpr BitBoard full() noexcept {
        BitBoard result;
        result.words_.fill(~Word{0});
        result.clear_unused_bits();
        return result;
    }

    [[nodiscard]]
    static constexpr BitBoard from_point(Point point) noexcept {
        BitBoard result;
        result.set(point);
        return result;
    }

    [[nodiscard]]
    static constexpr std::size_t index(Point point) noexcept {
        assert(point.x < BoardSize);
        assert(point.y < BoardSize);
        return static_cast<std::size_t>(point.y) * BoardSize + point.x;
    }

    [[nodiscard]]
    static constexpr Point point(std::size_t index) noexcept {
        assert(index < bit_count);
        return Point{static_cast<std::uint8_t>(index % BoardSize),
                     static_cast<std::uint8_t>(index / BoardSize)};
    }

    constexpr void set(Point point) noexcept { set(index(point)); }

    constexpr void set(std::size_t bit) noexcept {
        assert(bit < bit_count);
        words_[word_index(bit)] |= bit_mask(bit);
    }

    constexpr void reset(Point point) noexcept { reset(index(point)); }

    constexpr void reset(std::size_t bit) noexcept {
        assert(bit < bit_count);
        words_[word_index(bit)] &= ~bit_mask(bit);
    }

    constexpr void assign(Point point, bool value) noexcept { value ? set(point) : reset(point); }

    constexpr void clear() noexcept { words_.fill(0); }

    [[nodiscard]]
    constexpr bool test(Point point) const noexcept {
        return test(index(point));
    }

    [[nodiscard]]
    constexpr bool test(std::size_t bit) const noexcept {
        assert(bit < bit_count);
        return (words_[word_index(bit)] & bit_mask(bit)) != 0;
    }

    [[nodiscard]]
    constexpr bool any() const noexcept {
        for (Word word : words_) {
            if (word != 0) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]]
    constexpr bool none() const noexcept {
        return !any();
    }

    [[nodiscard]]
    constexpr std::size_t count() const noexcept {
        std::size_t result = 0;
        for (Word word : words_) {
            result += static_cast<std::size_t>(std::popcount(word));
        }
        return result;
    }

    [[nodiscard]]
    constexpr const Storage &words() const noexcept {
        return words_;
    }

    [[nodiscard]]
    constexpr Word word(std::size_t index) const noexcept {
        assert(index < word_count);
        return words_[index];
    }

    constexpr BitBoard &operator|=(const BitBoard &other) noexcept {
        for (std::size_t i = 0; i < word_count; ++i) {
            words_[i] |= other.words_[i];
        }
        return *this;
    }

    constexpr BitBoard &operator&=(const BitBoard &other) noexcept {
        for (std::size_t i = 0; i < word_count; ++i) {
            words_[i] &= other.words_[i];
        }
        return *this;
    }

    constexpr BitBoard &operator^=(const BitBoard &other) noexcept {
        for (std::size_t i = 0; i < word_count; ++i) {
            words_[i] ^= other.words_[i];
        }
        return *this;
    }

    [[nodiscard]]
    friend constexpr BitBoard operator|(BitBoard lhs, const BitBoard &rhs) noexcept {
        return lhs |= rhs;
    }

    [[nodiscard]]
    friend constexpr BitBoard operator&(BitBoard lhs, const BitBoard &rhs) noexcept {
        return lhs &= rhs;
    }

    [[nodiscard]]
    friend constexpr BitBoard operator^(BitBoard lhs, const BitBoard &rhs) noexcept {
        return lhs ^= rhs;
    }

    [[nodiscard]]
    friend constexpr BitBoard operator~(BitBoard board) noexcept {
        for (Word &word : board.words_) {
            word = ~word;
        }
        board.clear_unused_bits();
        return board;
    }

    [[nodiscard]]
    friend constexpr BitBoard operator-(BitBoard lhs, const BitBoard &rhs) noexcept {
        // Set difference.
        for (std::size_t i = 0; i < word_count; ++i) {
            lhs.words_[i] &= ~rhs.words_[i];
        }
        return lhs;
    }

    [[nodiscard]]
    friend constexpr bool operator==(const BitBoard &, const BitBoard &) noexcept = default;

    [[nodiscard]]
    constexpr bool intersects(const BitBoard &other) const noexcept {
        for (std::size_t i = 0; i < word_count; ++i) {
            if ((words_[i] & other.words_[i]) != 0) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]]
    constexpr bool contains(const BitBoard &other) const noexcept {
        for (std::size_t i = 0; i < word_count; ++i) {
            if ((words_[i] & other.words_[i]) != other.words_[i]) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]]
    constexpr std::size_t first_set_index() const noexcept {
        for (std::size_t i = 0; i < word_count; ++i) {
            if (words_[i] != 0) {
                return i * word_bits + static_cast<std::size_t>(std::countr_zero(words_[i]));
            }
        }
        return bit_count;
    }

    constexpr bool pop_first(Point &output) noexcept {
        for (std::size_t i = 0; i < word_count; ++i) {
            Word &word = words_[i];

            if (word == 0) {
                continue;
            }

            const unsigned offset = std::countr_zero(word);
            const std::size_t bit = i * word_bits + offset;

            word &= word - 1;
            output = point(bit);
            return true;
        }

        return false;
    }

    class SetBitIterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = Point;
        using difference_type = std::ptrdiff_t;

        constexpr SetBitIterator(const BitBoard *board, std::size_t word_index,
                                 Word remaining) noexcept
            : board_(board), word_index_(word_index), remaining_(remaining) {
            advance_to_nonempty_word();
        }

        [[nodiscard]]
        constexpr Point operator*() const noexcept {
            const std::size_t bit =
                word_index_ * word_bits + static_cast<std::size_t>(std::countr_zero(remaining_));

            return point(bit);
        }

        constexpr SetBitIterator &operator++() noexcept {
            remaining_ &= remaining_ - 1;

            if (remaining_ == 0) {
                ++word_index_;
                advance_to_nonempty_word();
            }

            return *this;
        }

        constexpr SetBitIterator operator++(int) noexcept {
            SetBitIterator previous = *this;
            ++(*this);
            return previous;
        }

        friend constexpr bool operator==(const SetBitIterator &lhs,
                                         const SetBitIterator &rhs) noexcept {
            return lhs.board_ == rhs.board_ && lhs.word_index_ == rhs.word_index_ &&
                   lhs.remaining_ == rhs.remaining_;
        }

    private:
        constexpr void advance_to_nonempty_word() noexcept {
            while (word_index_ < word_count) {
                remaining_ = board_->words_[word_index_];

                if (remaining_ != 0) {
                    return;
                }

                ++word_index_;
            }

            remaining_ = 0;
        }

        const BitBoard *board_;
        std::size_t word_index_;
        Word remaining_;
    };

    class SetBitsView {
    public:
        explicit constexpr SetBitsView(const BitBoard &board) noexcept : board_(board) {}

        [[nodiscard]]
        constexpr SetBitIterator begin() const noexcept {
            return SetBitIterator{&board_, 0, board_.words_[0]};
        }

        [[nodiscard]]
        constexpr SetBitIterator end() const noexcept {
            return SetBitIterator{&board_, word_count, 0};
        }

    private:
        const BitBoard &board_;
    };

    [[nodiscard]]
    constexpr SetBitsView set_bits() const noexcept {
        return SetBitsView{*this};
    }

private:
    [[nodiscard]]
    static constexpr std::size_t word_index(std::size_t bit) noexcept {
        return bit / word_bits;
    }

    [[nodiscard]]
    static constexpr Word bit_mask(std::size_t bit) noexcept {
        return Word{1} << (bit % word_bits);
    }

    [[nodiscard]]
    static constexpr Word final_word_mask() noexcept {
        constexpr std::size_t used_bits = bit_count % word_bits;

        if constexpr (used_bits == 0) {
            return ~Word{0};
        } else {
            return (Word{1} << used_bits) - 1;
        }
    }

    constexpr void clear_unused_bits() noexcept { words_.back() &= final_word_mask(); }

    Storage words_{};
};
