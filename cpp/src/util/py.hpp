#pragma once

#include <cmath>
#include <cstddef>
#include <iterator>
#include <tuple>
#include <utility>

//////////////////////////
// 1. Python-like range //
//////////////////////////

template <typename T = int> class range {
public:
    // Iterator type for range
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

        // For a forward range, we use a simple inequality test.
        bool operator!=(const Iterator &other) const {
            // For positive steps, continue while current_ < other.current_
            // For negative steps, continue while current_ > other.current_
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

//////////////////////////
// 2. Python-like zip   //
//////////////////////////

// Helper: Check if any iterator in tuple equals its corresponding end.
template <typename TupleIt, typename TupleEnd, std::size_t... I>
bool anyEqualImpl(const TupleIt &it, const TupleEnd &end, std::index_sequence<I...>) {
    return ((std::get<I>(it) == std::get<I>(end)) || ...);
}

template <typename TupleIt, typename TupleEnd>
bool anyEqual(const TupleIt &it, const TupleEnd &end) {
    return anyEqualImpl(it, end, std::make_index_sequence<std::tuple_size<TupleIt>::value>{});
}

namespace impl {
// zip_wrapper holds references to the provided containers.
template <typename... Containers> class ZipWrapper {
public:
    explicit ZipWrapper(Containers &...containers) : m_containers(std::tie(containers...)) {}

    // zip_iterator stores a tuple of current iterators and the corresponding ends.
    class Iterator {
    public:
        using iterator_tuple = std::tuple<decltype(std::begin(std::declval<Containers &>()))...>;

        Iterator(iterator_tuple current, iterator_tuple ends) : m_current(current), m_ends(ends) {}

        Iterator &operator++() {
            std::apply([](auto &...it) { ((++it), ...); }, m_current);
            return *this;
        }

        auto operator*() const {
            // Return a tuple of references to the elements.
            return std::apply([](auto &...it) { return std::tie(*it...); }, m_current);
        }

        bool operator!=(const Iterator &) const {
            // Stop if any iterator equals its end (like Python’s zip)
            return !anyEqual(m_current, m_ends);
        }

    private:
        iterator_tuple m_current;
        iterator_tuple m_ends;
    };

    Iterator begin() {
        auto beginTuple =
            std::apply([](auto &...container) { return std::make_tuple(std::begin(container)...); },
                       m_containers);
        auto endTuple =
            std::apply([](auto &...container) { return std::make_tuple(std::end(container)...); },
                       m_containers);
        return Iterator(beginTuple, endTuple);
    }

    Iterator end() {
        // The end iterator is defined as having all underlying iterators at their ends.
        auto endTuple =
            std::apply([](auto &...container) { return std::make_tuple(std::end(container)...); },
                       m_containers);
        return Iterator(endTuple, endTuple);
    }

private:
    std::tuple<Containers &...> m_containers;
};
} // namespace impl

// Helper function to deduce types.
template <typename... Containers> impl::ZipWrapper<Containers...> zip(Containers &...containers) {
    return impl::ZipWrapper<Containers...>(containers...);
}

////////////////////////////
// 3. Python-like enumerate //
////////////////////////////

namespace impl {

template <typename Container> class EnumerateWrapper {
public:
    explicit EnumerateWrapper(Container &container) : m_container(container) {}

    class Iterator {
    public:
        using inner_iterator = decltype(std::begin(std::declval<Container &>()));
        using value_type =
            std::pair<std::size_t, decltype(*std::begin(std::declval<Container &>()))>;

        Iterator(inner_iterator it, std::size_t index) : m_it(it), m_index(index) {}

        Iterator &operator++() {
            ++m_it;
            ++m_index;
            return *this;
        }

        auto operator*() const { return std::make_pair(m_index, *m_it); }

        bool operator!=(const Iterator &other) const { return m_it != other.m_it; }

    private:
        inner_iterator m_it;
        std::size_t m_index;
    };

    Iterator begin() { return Iterator(std::begin(m_container), 0); }
    Iterator end() { return Iterator(std::end(m_container), 0); }

private:
    Container &m_container;
};

} // namespace impl

template <typename Container> impl::EnumerateWrapper<Container> enumerate(Container &container) {
    return impl::EnumerateWrapper<Container>(container);
}

// reverse wrapper
template <typename Container> class ReverseWrapper {
public:
    explicit ReverseWrapper(Container &container) : m_container(container) {}

    auto begin() { return std::rbegin(m_container); }
    auto end() { return std::rend(m_container); }

private:
    Container &m_container;
};

template <typename Container> ReverseWrapper<Container> reverse(Container &container) {
    return ReverseWrapper<Container>(container);
}

// Generic sum function for any iterable container
template <typename Container> auto sum(const Container &container) {
    using ValueType = typename std::decay<decltype(*std::begin(container))>::type;
    ValueType total = ValueType();
    for (const auto &elem : container) {
        total += elem;
    }
    return total;
}

// Generic div function for any iterable container
template <typename Container> auto div(const Container &container, float divisor) {
    Container copy = container;
    for (auto &elem : copy) {
        elem /= divisor;
    }
    return copy;
}

// Generic pow function for any iterable container
template <typename Container> auto pow(const Container &container, float power) {
    Container copy = container;
    for (auto &elem : copy) {
        elem = std::pow(elem, power);
    }
    return copy;
}

// Generic argmax function for any iterable container
template <typename Container> size_t argmax(const Container &container) {
    using ValueType = typename std::decay<decltype(*std::begin(container))>::type;
    ValueType max = ValueType();
    size_t index = 0;
    for (const auto &[i, elem] : enumerate(container)) {
        if (elem > max) {
            max = elem;
            index = i;
        }
    }
    return index;
}

// Generic multinomial function for any iterable container
template <typename Container> size_t multinomial(const Container &container) {
    using ValueType = typename std::decay<decltype(*std::begin(container))>::type;
    ValueType total = sum(container);
    ValueType cumulative = ValueType();
    float random = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    for (const auto &[i, elem] : enumerate(container)) {
        cumulative += elem / total;
        if (random < cumulative) {
            return i;
        }
    }
    return container.size() - 1;
}

// Generic clamp function for any type
template <typename T> T clamp(T value, T min, T max) { return std::max(min, std::min(value, max)); }

template <typename T> T lerp(T a, T b, float t) { return (1 - t) * a + t * b; }
