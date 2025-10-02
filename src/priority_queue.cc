/*
 * Copyright 2025 Siddhant Biradar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// --- priority_queue.cc --- //

#include "priority_queue.h"
#include <algorithm> // for std::swap

void PriorityQueue::push(const value_type& value) {
    data.push_back(value);
    sift_up(data.size() - 1);
}

void PriorityQueue::pop() {
    if (data.empty()) return;
    std::swap(data.front(), data.back());
    data.pop_back();
    if (!data.empty())
        sift_down(0);
}

const PriorityQueue::value_type& PriorityQueue::top() const {
    return data.front();
}

bool PriorityQueue::empty() const noexcept {
    return data.empty();
}

size_t PriorityQueue::size() const noexcept {
    return data.size();
}

void PriorityQueue::reserve(size_t capacity) {
    data.reserve(capacity);
}

void PriorityQueue::sift_up(size_t idx) {
    while (idx > 0) {
        size_t parent = (idx - 1) >> 1;

        if (!(data[idx].first < data[parent].first))
            break;

        std::swap(data[parent], data[idx]);
        idx = parent;
    }
}

void PriorityQueue::sift_down(size_t idx) {
    size_t n = data.size();
    while (true) {
        size_t left = (idx << 1) + 1;
        size_t right = left + 1;
        size_t smallest = idx;

        if (left < n && data[left].first < data[smallest].first)
            smallest = left;
        if (right < n && data[right].first < data[smallest].first)
            smallest = right;

        if (smallest == idx)
            break;

        std::swap(data[idx], data[smallest]);
        idx = smallest;
    }
}

