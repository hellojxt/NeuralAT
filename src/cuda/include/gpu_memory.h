#pragma once
#include <common.h>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

NWOB_NAMESPACE_BEGIN

#define DEBUG_GUARD_SIZE 32

inline std::atomic<size_t> &total_n_bytes_allocated()
{
    static std::atomic<size_t> s_total_n_bytes_allocated{0};
    return s_total_n_bytes_allocated;
}

template <class T>
class GPUMemory
{
    private:
        T *m_data = nullptr;
        size_t m_size = 0;  // Number of elements

    public:
        GPUMemory() {}
        GPUMemory(size_t size) { resize(size); }
        // Don't permit copy assignment to prevent performance accidents.
        // Copy is permitted through an explicit copy constructor.
        GPUMemory<T> &operator=(const GPUMemory<T> &other) = delete;
        explicit GPUMemory(const GPUMemory<T> &other) { copy_from_device(other); }

        void check_guards() const
        {
#if DEBUG_GUARD_SIZE > 0
            if (!m_data)
                return;
            uint8_t buf[DEBUG_GUARD_SIZE];
            const uint8_t *rawptr = (const uint8_t *)m_data;
            cudaMemcpy(buf, rawptr - DEBUG_GUARD_SIZE, DEBUG_GUARD_SIZE, cudaMemcpyDeviceToHost);
            for (int i = 0; i < DEBUG_GUARD_SIZE; ++i)
                if (buf[i] != 0xff)
                {
                    printf("TRASH BEFORE BLOCK offset %d data %p, read 0x%02x expected 0xff!\n", i, m_data, buf[i]);
                    break;
                }
            cudaMemcpy(buf, rawptr + m_size * sizeof(T), DEBUG_GUARD_SIZE, cudaMemcpyDeviceToHost);
            for (int i = 0; i < DEBUG_GUARD_SIZE; ++i)
                if (buf[i] != 0xfe)
                {
                    printf("TRASH AFTER BLOCK offset %d data %p, read 0x%02x expected 0xfe!\n", i, m_data, buf[i]);
                    break;
                }
#endif
        }

        void allocate_memory(size_t n_bytes)
        {
            if (n_bytes == 0)
            {
                return;
            }
            uint8_t *rawptr = nullptr;
            CUDA_CHECK_THROW(cudaMalloc(&rawptr, n_bytes + DEBUG_GUARD_SIZE * 2));

#if DEBUG_GUARD_SIZE > 0
            CUDA_CHECK_THROW(cudaMemset(rawptr, 0xff, DEBUG_GUARD_SIZE));
            CUDA_CHECK_THROW(cudaMemset(rawptr + n_bytes + DEBUG_GUARD_SIZE, 0xfe, DEBUG_GUARD_SIZE));
#endif
            if (rawptr)
                rawptr += DEBUG_GUARD_SIZE;
            m_data = (T *)(rawptr);
            total_n_bytes_allocated() += n_bytes;
        }

        void free_memory()
        {
            if (!m_data)
            {
                return;
            }

            uint8_t *rawptr = (uint8_t *)m_data;
            if (rawptr)
                rawptr -= DEBUG_GUARD_SIZE;
            CUDA_CHECK_THROW(cudaFree(rawptr));

            total_n_bytes_allocated() -= get_bytes();

            m_data = nullptr;
            m_size = 0;
        }

        /// Frees memory again
        HOST_DEVICE ~GPUMemory()
        {
#ifndef __CUDA_ARCH__
            try
            {
                if (m_data)
                {
                    free_memory();
                }
            }
            catch (std::runtime_error error)
            {
                // Don't need to report on memory-free problems when the driver is shutting down.
                if (std::string{error.what()}.find("driver shutting down") == std::string::npos)
                {
                    std::cerr << "Could not free memory: " << error.what() << std::endl;
                }
            }
#endif
        }

        /** @name Resizing/enlargement
         *  @{
         */
        /// Resizes the array to the exact new size, even if it is already larger
        void resize(const size_t size)
        {
            if (m_size != size)
            {
                if (m_size)
                {
                    try
                    {
                        free_memory();
                    }
                    catch (std::runtime_error error)
                    {
                        throw std::runtime_error{std::string{"Could not free memory: "} + error.what()};
                    }
                }

                if (size > 0)
                {
                    try
                    {
                        allocate_memory(size * sizeof(T));
                    }
                    catch (std::runtime_error error)
                    {
                        throw std::runtime_error{std::string{"Could not allocate memory: "} + error.what()};
                    }
                }

                m_size = size;
            }
        }

        /// Enlarges the array if its size is smaller
        void enlarge(const size_t size)
        {
            if (size > m_size)
            {
                resize(size);
            }
        }
        /** @} */

        /** @name Memset
         *  @{
         */
        /// Sets the memory of the first num_elements to value
        void memset(const int value, const size_t num_elements, const size_t offset = 0)
        {
            if (num_elements + offset > m_size)
            {
                throw std::runtime_error{std::string{"Trying to memset "} + std::to_string(num_elements) +
                                         " elements, but memory size is " + std::to_string(m_size)};
            }

            CUDA_CHECK_THROW(cudaMemset(m_data + offset, value, num_elements * sizeof(T)));
        }

        /// Sets the memory of the all elements to value
        void memset(const int value) { memset(value, m_size); }
        /** @} */

        /** @name Copy operations
         *  @{
         */
        /// Copy data of num_elements from the raw pointer on the host
        void copy_from_host(const T *host_data, const size_t num_elements)
        {
            CUDA_CHECK_THROW(cudaMemcpy(data(), host_data, num_elements * sizeof(T), cudaMemcpyHostToDevice));
        }

        /// Copy num_elements from the host vector
        void copy_from_host(const std::vector<T> &data, const size_t num_elements)
        {
            if (data.size() < num_elements)
            {
                throw std::runtime_error{std::string{"Trying to copy "} + std::to_string(num_elements) +
                                         " elements, but vector size is " + std::to_string(data.size())};
            }
            copy_from_host(data.data(), num_elements);
        }

        /// Copies data from the raw host pointer to fill the entire array
        void copy_from_host(const T *data) { copy_from_host(data, m_size); }

        /// Copies num_elements of data from the raw host pointer after enlarging the array so that everything fits in
        void enlarge_and_copy_from_host(const T *data, const size_t num_elements)
        {
            enlarge(num_elements);
            copy_from_host(data, num_elements);
        }

        /// Copies num_elements from the host vector after enlarging the array so that everything fits in
        void enlarge_and_copy_from_host(const std::vector<T> &data, const size_t num_elements)
        {
            enlarge_and_copy_from_host(data.data(), num_elements);
        }

        /// Copies the entire host vector after enlarging the array so that everything fits in
        void enlarge_and_copy_from_host(const std::vector<T> &data)
        {
            enlarge_and_copy_from_host(data.data(), data.size());
        }

        /// Copies num_elements of data from the raw host pointer after resizing the array
        void resize_and_copy_from_host(const T *data, const size_t num_elements)
        {
            resize(num_elements);
            copy_from_host(data, num_elements);
        }

        /// Copies num_elements from the host vector after resizing the array
        void resize_and_copy_from_host(const std::vector<T> &data, const size_t num_elements)
        {
            resize_and_copy_from_host(data.data(), num_elements);
        }

        /// Copies the entire host vector after resizing the array
        void resize_and_copy_from_host(const std::vector<T> &data)
        {
            resize_and_copy_from_host(data.data(), data.size());
        }

        /// Copies the entire host vector to the device. Fails if there is not enough space available.
        void copy_from_host(const std::vector<T> &data)
        {
            if (data.size() < m_size)
            {
                throw std::runtime_error{std::string{"Trying to copy "} + std::to_string(m_size) +
                                         " elements, but vector size is " + std::to_string(data.size())};
            }
            copy_from_host(data.data(), m_size);
        }

        /// Copies num_elements of data from the raw host pointer to the device. Fails if there is not enough space
        /// available.
        void copy_to_host(T *host_data, const size_t num_elements) const
        {
            if (num_elements > m_size)
            {
                throw std::runtime_error{std::string{"Trying to copy "} + std::to_string(num_elements) +
                                         " elements, but memory size is " + std::to_string(m_size)};
            }

            CUDA_CHECK_THROW(cudaMemcpy(host_data, data(), num_elements * sizeof(T), cudaMemcpyDeviceToHost));
        }

        /// Copies num_elements from the device to a vector on the host
        void copy_to_host(std::vector<T> &data, const size_t num_elements) const
        {
            if (data.size() < num_elements)
            {
                throw std::runtime_error{std::string{"Trying to copy "} + std::to_string(num_elements) +
                                         " elements, but vector size is " + std::to_string(data.size())};
            }

            copy_to_host(data.data(), num_elements);
        }

        /// Copies num_elements from the device to a raw pointer on the host
        void copy_to_host(T *data) const { copy_to_host(data, m_size); }

        /// Copies all elements from the device to a vector on the host
        void copy_to_host(std::vector<T> &data) const
        {
            if (data.size() < m_size)
            {
                throw std::runtime_error{std::string{"Trying to copy "} + std::to_string(m_size) +
                                         " elements, but vector size is " + std::to_string(data.size())};
            }

            copy_to_host(data.data(), m_size);
        }

        /// Copies size elements from another device array to this one, automatically resizing it
        void copy_from_device(const GPUMemory<T> &other, const size_t size)
        {
            if (size == 0)
            {
                return;
            }

            if (m_size < size)
            {
                resize(size);
            }

            CUDA_CHECK_THROW(cudaMemcpy(m_data, other.m_data, size * sizeof(T), cudaMemcpyDeviceToDevice));
        }

        /// Copies data from another device array to this one, automatically resizing it
        void copy_from_device(const GPUMemory<T> &other) { copy_from_device(other, other.m_size); }

        // Created an (owned) copy of the data
        GPUMemory<T> copy(size_t size) const
        {
            GPUMemory<T> result{size};
            result.copy_from_device(*this);
            return result;
        }

        GPUMemory<T> copy() const { return copy(m_size); }

        T *data() const
        {
            check_guards();
            return m_data;
        }

        T *begin() const { return data(); }
        T *end() const { return data() + m_size; }

        size_t get_num_elements() const { return m_size; }

        size_t size() const { return get_num_elements(); }

        size_t get_bytes() const { return m_size * sizeof(T); }

        size_t bytes() const { return get_bytes(); }

        T *device_ptr() const { return data(); }
};

template <typename T, size_t N>
struct PitchedPtr
{
        HOST_DEVICE PitchedPtr() : ptr(nullptr) {}

        template <typename... Sizes>
        HOST_DEVICE PitchedPtr(T *ptr, Sizes... sizes) : ptr(ptr)
        {
            set(ptr, sizes...);
        }

        template <typename... Sizes>
        HOST_DEVICE void set(T *ptr, Sizes... sizes)
        {
            static_assert(sizeof...(Sizes) == N, "Wrong number of sizes");
            size_t sizes_array[N] = {static_cast<size_t>(sizes)...};
            size[N - 1] = sizes_array[N - 1];
            stride[N - 1] = 1;
#pragma unroll
            for (int i = N - 2; i >= 0; --i)
            {
                size[i] = sizes_array[i];
                stride[i] = stride[i + 1] * size[i + 1];
            }
            this->ptr = ptr;
        }

        template <typename... Indices>
        HOST_DEVICE T &operator()(Indices... indices) const
        {
            static_assert(sizeof...(Indices) == N, "Wrong number of indices");
            return ptr[get_index(indices...)];
        }

        HOST_DEVICE T &operator()(int3 coord) const
        {
            static_assert(N == 3, "int3 operator can only be used with N=3");
            return ptr[get_index(coord.x, coord.y, coord.z)];
        }

        template <typename... Indices>
        HOST_DEVICE size_t get_index(Indices... indices) const
        {
            size_t indices_array[N] = {static_cast<size_t>(indices)...};
            size_t index = 0;
#pragma unroll
            for (int i = 0; i < N; ++i)
            {
                index += indices_array[i] * stride[i];
            }
            return index;
        }

        T *ptr;
        size_t stride[N];
        size_t size[N];
};

NWOB_NAMESPACE_END