#pragma once

#include <cstddef>
#include <stdexcept>

template<typename T>
class ChunkedList {
private:
    struct Node {
        T* data;
        std::size_t capacity;
        std::size_t size;
        Node* next;

        Node(std::size_t cap)
            : data(new T[cap]), capacity(cap), size(0), next(nullptr) {}

        // Disable copy constructor and assignment operator
        Node(const Node&) = delete;
        Node& operator=(const Node&) = delete;

        ~Node() { delete[] data; }
    };

    Node* head;
    Node* tail;
    std::size_t count;

public:
    ChunkedList()
        : head(nullptr), tail(nullptr), count(0) {}

    ~ChunkedList() {
        clear();
    }

    // Add an element; returns reference so pointer remains valid
    T& add(const T& value) {
        if (!tail) {
            head = tail = new Node(2);
        } else if (tail->size == tail->capacity) {
            std::size_t newCap = tail->capacity * 2;
            tail->next = new Node(newCap);
            tail = tail->next;
        }
        tail->data[tail->size] = value;
        ++tail->size;
        ++count;
        return tail->data[tail->size - 1];
    }

    // Access by index (0-based)
    T& operator[](std::size_t index) {
        if (index >= count) {
            throw std::out_of_range("Index out of range");
        }
        Node* node = head;
        while (node) {
            if (index < node->size) {
                return node->data[index];
            }
            index -= node->size;
            node = node->next;
        }
        // Should not reach here
        throw std::out_of_range("Index out of range");
    }

    std::size_t size() const { return count; }

    void clear() {
        Node* node = head;
        while (node) {
            Node* tmp = node->next;
            delete node;
            node = tmp;
        }
        head = tail = nullptr;
        count = 0;
    }
};
