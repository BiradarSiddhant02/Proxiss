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

#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <random>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "distance.hpp"
#include "proxi_flat.h"

template <typename T> class RandomGenerator {
public:
    using Dist = std::conditional_t<std::is_integral<T>::value, std::uniform_int_distribution<T>,
                                    std::uniform_real_distribution<T>>;

    RandomGenerator(T min, T max) : dist(min, max), rng(std::random_device{}()) {}

    T operator()() { return dist(rng); }

private:
    Dist dist;
    std::mt19937 rng;
};

class ProxiCluster {
private:



public:
};