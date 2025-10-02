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

// --- priority_queue.h --- //

#pragma once
#include <cstddef>
#include <utility>
#include <vector>

class PriorityQueue {
public:
    using value_type = std::pair<float, size_t>;

    PriorityQueue() = default;

    void push(const value_type& value);
    void pop();
    const value_type& top() const;
    bool empty() const noexcept;
    size_t size() const noexcept;
    void reserve(size_t capacity);

private:
    std::vector<value_type> data;

    void sift_up(size_t idx);
    void sift_down(size_t idx);
};

