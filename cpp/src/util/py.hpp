#pragma once

#include <cstddef>
#include <iterator>

// Python-like range; the rest of the former Python-emulation helpers were unused and removed.
template <typename T = int> class range {
public:
    class Iterator {
    public:
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

        Iterator(T current, T step) : m_current(current), m_step(step) {}

        T operator*() const { return m_current; }

        Iterator &operator++() {
            m_current += m_step;
            return *this;
        }

        bool operator!=(const Iterator &other) const {
            return m_step > 0 ? (m_current < other.m_current) : (m_current > other.m_current);
        }

    private:
        T m_current;
        T m_step;
    };

    range(T stop) : range(T(0), stop, T(1)) {}
    range(T start, T stop) : range(start, stop, T(1)) {}
    range(T start, T stop, T step) : m_start(start), m_stop(stop), m_step(step) {}

    Iterator begin() const { return Iterator(m_start, m_step); }
    Iterator end() const { return Iterator(m_stop, m_step); }

private:
    T m_start, m_stop, m_step;
};
