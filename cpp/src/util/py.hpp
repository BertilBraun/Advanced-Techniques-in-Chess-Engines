#pragma once

#include <cstddef>
#include <tuple>
#include <iterator>
#include <utility>

//////////////////////////
// 1. Python-like range //
//////////////////////////

template<typename T = int>
class range {
public:
    // Iterator type for range
    class iterator {
    public:
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;

        iterator(T current, T step) : current_(current), step_(step) {}

        T operator*() const { return current_; }

        iterator& operator++() { 
            current_ += step_;
            return *this;
        }

        // For a forward range, we use a simple inequality test.
        bool operator!=(const iterator& other) const {
            // For positive steps, continue while current_ < other.current_
            // For negative steps, continue while current_ > other.current_
            return step_ > 0 ? (current_ < other.current_) 
                             : (current_ > other.current_);
        }

    private:
        T current_;
        T step_;
    };

    range(T stop) : range(T(0), stop, T(1)) { }
    range(T start, T stop) : range(start, stop, T(1)) { }
    range(T start, T stop, T step) : start_(start), stop_(stop), step_(step) { }

    iterator begin() const { return iterator(start_, step_); }
    iterator end()   const { return iterator(stop_, step_); }

private:
    T start_, stop_, step_;
};

//////////////////////////
// 2. Python-like zip   //
//////////////////////////

// Helper: Check if any iterator in tuple equals its corresponding end.
template<typename TupleIt, typename TupleEnd, std::size_t... I>
bool any_equal_impl(const TupleIt& it, const TupleEnd& end, std::index_sequence<I...>) {
    return ((std::get<I>(it) == std::get<I>(end)) || ...);
}

template<typename TupleIt, typename TupleEnd>
bool any_equal(const TupleIt& it, const TupleEnd& end) {
    return any_equal_impl(it, end, std::make_index_sequence<std::tuple_size<TupleIt>::value>{});
}

namespace impl {
// zip_wrapper holds references to the provided containers.
template <typename... Containers>
class zip_wrapper {
public:
    explicit zip_wrapper(Containers&... containers)
        : containers_(std::tie(containers...)) {}

    // zip_iterator stores a tuple of current iterators and the corresponding ends.
    class iterator {
    public:
        using iterator_tuple = std::tuple<decltype(std::begin(std::declval<Containers&>()))...>;

        iterator(iterator_tuple current, iterator_tuple ends)
            : current_(current), ends_(ends) { }

        iterator& operator++() {
            std::apply([](auto&... it){ ((++it), ...); }, current_);
            return *this;
        }

        auto operator*() const {
            // Return a tuple of references to the elements.
            return std::apply([](auto&... it){ return std::tie(*it...); }, current_);
        }

        bool operator!=(const iterator&) const {
            // Stop if any iterator equals its end (like Python’s zip)
            return !any_equal(current_, ends_);
        }

    private:
        iterator_tuple current_;
        iterator_tuple ends_;
    };

    iterator begin() {
        auto begin_tuple = std::apply([](auto&... container) {
            return std::make_tuple(std::begin(container)...);
        }, containers_);
        auto end_tuple = std::apply([](auto&... container) {
            return std::make_tuple(std::end(container)...);
        }, containers_);
        return iterator(begin_tuple, end_tuple);
    }

    iterator end() {
        // The end iterator is defined as having all underlying iterators at their ends.
        auto end_tuple = std::apply([](auto&... container) {
            return std::make_tuple(std::end(container)...);
        }, containers_);
        return iterator(end_tuple, end_tuple);
    }

private:
    std::tuple<Containers&...> containers_;
};
}

// Helper function to deduce types.
template <typename... Containers>
impl::zip_wrapper<Containers...> zip(Containers&... containers) {
    return impl::zip_wrapper<Containers...>(containers...);
}

////////////////////////////
// 3. Python-like enumerate //
////////////////////////////

namespace impl {

template <typename Container>
class enumerate_wrapper {
public:
    explicit enumerate_wrapper(Container& container) : container_(container) {}

    class iterator {
    public:
        using inner_iterator = decltype(std::begin(std::declval<Container&>()));
        using value_type = std::pair<std::size_t, decltype(*std::begin(std::declval<Container&>()))>;

        iterator(inner_iterator it, std::size_t index) : it_(it), index_(index) { }

        iterator& operator++() { 
            ++it_;
            ++index_;
            return *this;
        }

        auto operator*() const {
            return std::make_pair(index_, *it_);
        }

        bool operator!=(const iterator& other) const {
            return it_ != other.it_;
        }

    private:
        inner_iterator it_;
        std::size_t index_;
    };

    iterator begin() { return iterator(std::begin(container_), 0); }
    iterator end()   { return iterator(std::end(container_), 0); }

private:
    Container& container_;
};

}

template <typename Container>
impl::enumerate_wrapper<Container> enumerate(Container& container) {
    return impl::enumerate_wrapper<Container>(container);
}
