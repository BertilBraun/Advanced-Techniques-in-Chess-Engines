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
    static constexpr std::size_t bitCount = BoardSize * BoardSize;
    static constexpr std::size_t wordBits = 64;
    static constexpr std::size_t wordCount = (bitCount + wordBits - 1) / wordBits;

    using Word = std::uint64_t;
    using Storage = std::array<Word, wordCount>;

    struct Point {
        std::uint8_t x;
        std::uint8_t y;

        friend constexpr bool operator==(Point, Point) = default;
    };

    constexpr BitBoard() noexcept = default;

    explicit constexpr BitBoard(Storage words) noexcept : m_words(words) { clearUnusedBits(); }

    [[nodiscard]]
    static constexpr BitBoard full() noexcept {
        BitBoard result;
        result.m_words.fill(~Word{0});
        result.clearUnusedBits();
        return result;
    }

    [[nodiscard]]
    static constexpr BitBoard fromPoint(Point point) noexcept {
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
        assert(index < bitCount);
        return Point{
            .x = static_cast<std::uint8_t>(index % BoardSize),
            .y = static_cast<std::uint8_t>(index / BoardSize),
        };
    }

    constexpr void set(Point point) noexcept { set(index(point)); }

    constexpr void set(std::size_t bit) noexcept {
        assert(bit < bitCount);
        m_words[wordIndex(bit)] |= bitMask(bit);
    }

    constexpr void reset(Point point) noexcept { reset(index(point)); }

    constexpr void reset(std::size_t bit) noexcept {
        assert(bit < bitCount);
        m_words[wordIndex(bit)] &= ~bitMask(bit);
    }

    constexpr void assign(Point point, bool value) noexcept { value ? set(point) : reset(point); }

    constexpr void clear() noexcept { m_words.fill(0); }

    [[nodiscard]]
    constexpr bool test(Point point) const noexcept {
        return test(index(point));
    }

    [[nodiscard]]
    constexpr bool test(std::size_t bit) const noexcept {
        assert(bit < bitCount);
        return (m_words[wordIndex(bit)] & bitMask(bit)) != 0;
    }

    [[nodiscard]]
    constexpr bool any() const noexcept {
        for (Word word : m_words) {
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
        for (Word word : m_words) {
            result += static_cast<std::size_t>(std::popcount(word));
        }
        return result;
    }

    [[nodiscard]]
    constexpr const Storage &words() const noexcept {
        return m_words;
    }

    [[nodiscard]]
    constexpr Word word(std::size_t index) const noexcept {
        assert(index < wordCount);
        return m_words[index];
    }

    constexpr BitBoard &operator|=(const BitBoard &other) noexcept {
        for (std::size_t i = 0; i < wordCount; ++i) {
            m_words[i] |= other.m_words[i];
        }
        return *this;
    }

    constexpr BitBoard &operator&=(const BitBoard &other) noexcept {
        for (std::size_t i = 0; i < wordCount; ++i) {
            m_words[i] &= other.m_words[i];
        }
        return *this;
    }

    constexpr BitBoard &operator^=(const BitBoard &other) noexcept {
        for (std::size_t i = 0; i < wordCount; ++i) {
            m_words[i] ^= other.m_words[i];
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
        for (Word &word : board.m_words) {
            word = ~word;
        }
        board.clearUnusedBits();
        return board;
    }

    [[nodiscard]]
    friend constexpr BitBoard operator-(BitBoard lhs, const BitBoard &rhs) noexcept {
        // Set difference.
        for (std::size_t i = 0; i < wordCount; ++i) {
            lhs.m_words[i] &= ~rhs.m_words[i];
        }
        return lhs;
    }

    [[nodiscard]]
    friend constexpr bool operator==(const BitBoard &, const BitBoard &) noexcept = default;

    [[nodiscard]]
    constexpr bool intersects(const BitBoard &other) const noexcept {
        for (std::size_t i = 0; i < wordCount; ++i) {
            if ((m_words[i] & other.m_words[i]) != 0) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]]
    constexpr bool contains(const BitBoard &other) const noexcept {
        for (std::size_t i = 0; i < wordCount; ++i) {
            if ((m_words[i] & other.m_words[i]) != other.m_words[i]) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]]
    constexpr std::size_t firstSetIndex() const noexcept {
        for (std::size_t i = 0; i < wordCount; ++i) {
            if (m_words[i] != 0) {
                return i * wordBits + static_cast<std::size_t>(std::countr_zero(m_words[i]));
            }
        }
        return bitCount;
    }

    constexpr bool popFirst(Point &output) noexcept {
        for (std::size_t i = 0; i < wordCount; ++i) {
            Word &word = m_words[i];

            if (word == 0) {
                continue;
            }

            const unsigned offset = std::countr_zero(word);
            const std::size_t bit = i * wordBits + offset;

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

        constexpr SetBitIterator(const BitBoard *board, std::size_t startWordIndex,
                                 Word remaining) noexcept
            : m_board(board), m_wordIndex(startWordIndex), m_remaining(remaining) {
            advanceToNonemptyWord();
        }

        [[nodiscard]]
        constexpr Point operator*() const noexcept {
            const std::size_t bit =
                m_wordIndex * wordBits + static_cast<std::size_t>(std::countr_zero(m_remaining));

            return point(bit);
        }

        constexpr SetBitIterator &operator++() noexcept {
            m_remaining &= m_remaining - 1;

            if (m_remaining == 0) {
                ++m_wordIndex;
                advanceToNonemptyWord();
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
            return lhs.m_board == rhs.m_board && lhs.m_wordIndex == rhs.m_wordIndex &&
                   lhs.m_remaining == rhs.m_remaining;
        }

    private:
        constexpr void advanceToNonemptyWord() noexcept {
            while (m_wordIndex < wordCount) {
                m_remaining = m_board->m_words[m_wordIndex];

                if (m_remaining != 0) {
                    return;
                }

                ++m_wordIndex;
            }

            m_remaining = 0;
        }

        const BitBoard *m_board;
        std::size_t m_wordIndex;
        Word m_remaining;
    };

    class SetBitsView {
    public:
        explicit constexpr SetBitsView(const BitBoard &board) noexcept : m_board(board) {}

        [[nodiscard]]
        constexpr SetBitIterator begin() const noexcept {
            return SetBitIterator{&m_board, 0, m_board.m_words[0]};
        }

        [[nodiscard]]
        constexpr SetBitIterator end() const noexcept {
            return SetBitIterator{&m_board, wordCount, 0};
        }

    private:
        const BitBoard &m_board;
    };

    [[nodiscard]]
    constexpr SetBitsView setBits() const noexcept {
        return SetBitsView{*this};
    }

private:
    [[nodiscard]]
    static constexpr std::size_t wordIndex(std::size_t bit) noexcept {
        return bit / wordBits;
    }

    [[nodiscard]]
    static constexpr Word bitMask(std::size_t bit) noexcept {
        return Word{1} << (bit % wordBits);
    }

    [[nodiscard]]
    static constexpr Word finalWordMask() noexcept {
        constexpr std::size_t usedBits = bitCount % wordBits;

        if constexpr (usedBits == 0) {
            return ~Word{0};
        } else {
            return (Word{1} << usedBits) - 1;
        }
    }

    constexpr void clearUnusedBits() noexcept { m_words.back() &= finalWordMask(); }

    Storage m_words{};
};
