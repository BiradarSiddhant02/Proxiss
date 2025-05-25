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

// --- mm_file.hpp --- //

#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

class MemoryMappedFile {
public:
    MemoryMappedFile(const std::string &filename, size_t size)
        : m_filename(filename), m_size(size), m_data_raw(nullptr),
#ifdef _WIN32
          m_fileHandle(INVALID_HANDLE_VALUE), m_mapHandle(NULL),
#else
          m_fd(-1),
#endif
          m_data(nullptr, [](std::byte *) {}) {

#ifdef _WIN32
        m_fileHandle = CreateFileA(m_filename.c_str(), GENERIC_READ | GENERIC_WRITE,
                                   FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_ALWAYS,
                                   FILE_ATTRIBUTE_NORMAL, NULL);
        if (m_fileHandle == INVALID_HANDLE_VALUE)
            throw std::runtime_error("Failed to open file");

        m_mapHandle = CreateFileMappingA(m_fileHandle, NULL, PAGE_READWRITE, 0, m_size, NULL);
        if (m_mapHandle == NULL) {
            CloseHandle(m_fileHandle);
            throw std::runtime_error("Failed to create file mapping");
        }

        m_data_raw = MapViewOfFile(m_mapHandle, FILE_MAP_ALL_ACCESS, 0, 0, m_size);
        if (m_data_raw == NULL) {
            CloseHandle(m_mapHandle);
            CloseHandle(m_fileHandle);
            throw std::runtime_error("Failed to map file");
        }
#else
        // Open or create the file
        m_fd = open(m_filename.c_str(), O_RDWR | O_CREAT, 0666);
        if (m_fd < 0)
            throw std::runtime_error("Failed to open file");

        // Resize the file to the requested size
        if (ftruncate(m_fd, m_size) == -1) {
            close(m_fd);
            throw std::runtime_error("Failed to resize file");
        }

        // Memory-map the file
        m_data_raw = mmap(nullptr, m_size, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, 0);
        if (m_data_raw == MAP_FAILED) {
            close(m_fd);
            throw std::runtime_error("Failed to map file");
        }
#endif
        // Wrap the memory in a unique_ptr with a no-op deleter
        m_data.reset(static_cast<std::byte *>(m_data_raw));
    }

    ~MemoryMappedFile() {
#ifdef _WIN32
        if (m_data_raw)
            UnmapViewOfFile(m_data_raw);
        if (m_mapHandle)
            CloseHandle(m_mapHandle);
        if (m_fileHandle != INVALID_HANDLE_VALUE)
            CloseHandle(m_fileHandle);
#else
        if (m_data_raw && m_data_raw != MAP_FAILED)
            munmap(m_data_raw, m_size);
        if (m_fd != -1)
            close(m_fd);
#endif
    }

    std::vector<float> read_embeddings(size_t offset = 0, size_t count = 0) const {
        // offset is in floats, not bytes
        size_t float_offset_bytes = offset * sizeof(float);
        if (float_offset_bytes >= m_size)
            throw std::out_of_range("Offset out of range");

        size_t maxFloats = (m_size - float_offset_bytes) / sizeof(float);
        size_t numFloats = (count == 0 || count > maxFloats) ? maxFloats : count;

        std::vector<float> result(numFloats);
        std::memcpy(result.data(), m_data.get() + float_offset_bytes, numFloats * sizeof(float));
        return result;
    }

private:
    std::string m_filename;
    size_t m_size;
    void *m_data_raw;
#ifdef _WIN32
    HANDLE m_fileHandle;
    HANDLE m_mapHandle;
#else
    int m_fd;
#endif
    std::unique_ptr<std::byte[], void (*)(std::byte *)> m_data;
};