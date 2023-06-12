#pragma once
#include <cassert>
#include <iostream>
#include <memory>
#include <vector>
#include "macro.h"
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <fstream>
#include <assert.h>

namespace GNNBEM
{

class to_cpu
{
    public:
        unsigned int x, y, z;
        to_cpu(int3 idx) : x(idx.x), y(idx.y), z(idx.z) {}
        to_cpu(int x, int y, int z) : x(x), y(y), z(z) {}
        to_cpu(uint3 idx) : x(idx.x), y(idx.y), z(idx.z) {}
        to_cpu(unsigned int x, unsigned int y, unsigned int z) : x(x), y(y), z(z) {}
        to_cpu(unsigned int x, unsigned int y) : x(x), y(y), z(0) {}
        to_cpu(unsigned int x) : x(x), y(0), z(0) {}
        to_cpu() : x(0), y(0), z(0) {}
};

/**
 * @brief Circular array
 */
template <typename T, unsigned int N>
class CircularArray
{
    private:
        T m_data[N];

    public:
        CGPU_FUNC CircularArray() {}
        CGPU_FUNC inline T &operator[](int id) { return m_data[(id + N) % N]; }
        CGPU_FUNC inline const T &operator[](int id) const { return m_data[(id + N) % N]; }
        void reset()
        {
            for (int i = 0; i < N; i++)
            {
                m_data[i] = 0;
            }
        }
        friend std::ostream &operator<<(std::ostream &out, const CircularArray &h)
        {
            for (int i = 0; i < N; i++)
                out << h[i] << " ";
            return out;
        }

        friend bool operator==(const CircularArray &a, const CircularArray &b)
        {
            for (int i = 0; i < N; i++)
            {
                if (a[i] != b[i])
                    return false;
            }
            return true;
        }
        CGPU_FUNC void print()
        {
            for (int i = 0; i < N; i++)
                printf("%f ", m_data[i]);
            printf("\n");
        }
};

template <typename T>
class CArr;
template <typename T>
class GArr;

template <typename Index>
class CompactIndexArray
{
    public:
        GArr<Index> data;
        int non_zero_size;
        CompactIndexArray() {}
        void reserve(int size_)
        {
            data.resize(size_);
            non_zero_size = 0;
        }
        struct is_zero
        {
                CGPU_FUNC bool operator()(const Index &x) const { return x.is_zero(); }
        };
        void remove_zeros() 
        {
            // 下面这个函数导致了400M的内存占用！并且无法被data.clear()释放！
            // 第一次调用thrust库会导致巨量内存泄漏？
            Index *new_end = thrust::remove_if(thrust::device, data.begin(), data.end(), is_zero());
            non_zero_size = new_end - data.begin();
        }

        void sort() { thrust::sort(thrust::device, data.begin(), data.begin() + non_zero_size); }

        GPU_FUNC Index &operator[](int i) { return data[i]; }
        CGPU_FUNC int size() const { return non_zero_size; }
        void clear()
        {
            non_zero_size = 0;
            data.clear();
        }
        void reset()
        {
            non_zero_size = 0;
            data.reset();
        }
};

template <typename T>
class CArr
{
    public:
        CArr(){};
        CArr(std::vector<T> &x) { m_data = x; };
        CArr(uint num) { m_data.resize((size_t)num); }
        CArr(T *x, uint num)
        {
            m_data.resize((size_t)num);
            for (uint i = 0; i < num; i++)
            {
                m_data[i] = x[i];
            }
        }
        CArr(const GArr<T> &v) { this->assign(v); }
        CArr(const GArr<T> &v, int offset, int count) { this->assign(v, count, 0, offset); }

        ~CArr(){};

        void resize(uint n);

        /*!
         *	\brief	Clear all data to zero.
         */
        void reset();
        void clear();

        inline const T *begin() const { return m_data.size() == 0 ? nullptr : &m_data[0]; }
        inline T *begin() { return m_data.size() == 0 ? nullptr : &m_data[0]; }
        inline const T *end() const { return m_data.size() == 0 ? nullptr : &m_data[0] + m_data.size(); }
        inline T *end() { return m_data.size() == 0 ? nullptr : &m_data[0] + m_data.size(); }

        inline const T *data() const { return m_data.size() == 0 ? nullptr : &m_data[0]; }
        inline T *data() { return m_data.size() == 0 ? nullptr : &m_data[0]; }

        inline T &operator[](unsigned int id) { return m_data[id]; }

        inline const T &operator[](unsigned int id) const { return m_data[id]; }

        inline uint size() const { return (uint)m_data.size(); }
        inline bool isEmpty() const { return m_data.empty(); }

        inline void pushBack(T ele) { m_data.push_back(ele); }

        inline T &last() { return m_data.back(); }

        void assign(const T &val);
        void assign(uint num, const T &val);
        void assign(const GArr<T> &src);
        void assign(const CArr<T> &src);
        void assign(const GArr<T> &src, const uint count, const uint dstOffset = 0, const uint srcOffset = 0);
        void assign(const CArr<T> &src, const uint count, const uint dstOffset = 0, const uint srcOffset = 0);
        void assign(const std::vector<T> &src, const uint count, const uint dstOffset = 0, const uint srcOffset = 0);

        friend std::ostream &operator<<(std::ostream &out, const CArr<T> &cArray)
        {
            for (uint i = 0; i < cArray.size(); i++)
            {
                out << i << ": " << cArray[i] << std::endl;
            }

            return out;
        }

#define MAX_FS_SIZE 1000
        friend std::ofstream &operator<<(std::ofstream &out, const CArr<T> &cArray)
        {
            auto size_new = min(cArray.size(), MAX_FS_SIZE);
            for (uint i = 0; i < size_new; i++)
            {
                out << cArray[i] << std::endl;
            }

            return out;
        }

        inline GArr<T> gpu() { return GArr<T>(*this); }

        inline T sum() const
        {
            T sum = 0;
            for (uint i = 0; i < m_data.size(); i++)
            {
                sum += m_data[i];
            }
            return sum;
        }

        inline std::vector<T> vector() { return m_data; }

    public:
        std::vector<T> m_data;
};

/*!
 *	\class	Array
 *	\brief	This class is designed to be elegant, so it can be directly passed to GPU as parameters.
 */
template <typename T>
class GArr
{
    public:
        GArr() { resize(0); };

        GArr(uint num) { this->resize(num); }

        GArr(T *data_, uint num)
        {
            this->m_data = data_;
            this->m_totalNum = num;
        }

        GArr(const CArr<T> &v) { this->assign(v); }

        /*!
         *	\brief	Do not release memory here, call clear() explicitly.
         */
        ~GArr(){};

        void resize(const uint n);

        /*!
         *	\brief	Clear all data to zero.
         */
        void reset();
        void reset_minus_one();
        /*!
         *	\brief	Free allocated memory.	Should be called before the object is deleted.
         */
        void clear();

        void remove(const T &val) { m_totalNum = thrust::remove(m_data, m_data + m_totalNum, val) - m_data; }

        CGPU_FUNC inline const T *begin() const { return m_data; }
        CGPU_FUNC inline T *begin() { return m_data; }

        CGPU_FUNC inline const T *data() const { return m_data; }
        CGPU_FUNC inline T *data() { return m_data; }

        CGPU_FUNC inline const T *end() const { return m_data + m_totalNum; }
        CGPU_FUNC inline T *end() { return m_data + m_totalNum; }

        GPU_FUNC inline T &operator[](unsigned int id)
        {
#ifdef MEMORY_CHECK
            assert(id < m_totalNum);
#endif
            return m_data[id];
        }

        GPU_FUNC inline T &operator[](unsigned int id) const
        {
#ifdef MEMORY_CHECK
            assert(id < m_totalNum);
#endif
            return m_data[id];
        }

        CPU_FUNC inline T operator[](to_cpu idx) const
        {
            T value;
            cudaMemcpy(&value, m_data + idx.x, sizeof(T), cudaMemcpyDeviceToHost);
            return value;
        }

        CPU_FUNC inline T operator[](to_cpu idx)
        {
            T value;
            cudaMemcpy(&value, m_data + idx.x, sizeof(T), cudaMemcpyDeviceToHost);
            return value;
        }

        CGPU_FUNC inline uint size() const { return m_totalNum; }
        CGPU_FUNC inline bool isCPU() const { return false; }
        CGPU_FUNC inline bool isGPU() const { return true; }
        CGPU_FUNC inline bool isEmpty() const { return m_data == nullptr; }

        void assign(const GArr<T> &src);
        void assign(const CArr<T> &src);
        void assign(const std::vector<T> &src);
        // GArr &operator=(const CArr<T> &v)
        // { this->assign(v); return *this; }

        inline T last_item()
        {
            T value;
            cudaMemcpy(&value, m_data + m_totalNum - 1, sizeof(T), cudaMemcpyDeviceToHost);
            return value;
        }

        inline CArr<T> cpu() { return CArr<T>(*this); }

        void assign(const GArr<T> &src, const uint count, const uint dstOffset = 0, const uint srcOffset = 0);
        void assign(const CArr<T> &src, const uint count, const uint dstOffset = 0, const uint srcOffset = 0);
        void assign(const std::vector<T> &src, const uint count, const uint dstOffset = 0, const uint srcOffset = 0);
        void copy_from(const GArr<T> &src)
        {
            if (src.size() != m_totalNum)
            {
                LOG_ERROR("Error in GArr.copy_from: size not match, " << src.size() << " != " << m_totalNum);
                return;
            }
            cuSafeCall(cudaMemcpy(m_data, src.begin(), m_totalNum * sizeof(T), cudaMemcpyDeviceToDevice));
        }

        friend std::ostream &operator<<(std::ostream &out, const GArr<T> &dArray)
        {
            CArr<T> hArray;
            hArray.assign(dArray);

            out << hArray;

            return out;
        }

        friend std::ofstream &operator<<(std::ofstream &out, const GArr<T> &dArray)
        {
            CArr<T> hArray;
            hArray.assign(dArray);
            out << hArray;
            return out;
        }

    private:
        T *m_data = 0;
        uint m_totalNum = 0;
};

}  // namespace GNNBEM

namespace GNNBEM
{
template <typename T>
void GArr<T>::resize(const uint n)
{
    //		assert(n >= 1);
    if (m_data != nullptr)
        clear();

    m_totalNum = n;
    if (n == 0)
    {
        m_data = nullptr;
    }
    else
        cuSafeCall(cudaMalloc(&m_data, n * sizeof(T)));
}

template <typename T>
void GArr<T>::clear()
{
    if (m_data != NULL)
    {
        cuSafeCall(cudaFree((void *)m_data));
    }

    m_data = NULL;
    m_totalNum = 0;
}

template <typename T>
void GArr<T>::reset()
{
    cuSafeCall(cudaMemset((void *)m_data, 0, m_totalNum * sizeof(T)));
}

template <typename T>
void GArr<T>::reset_minus_one()
{
    cuSafeCall(cudaMemset((void *)m_data, -1, m_totalNum * sizeof(T)));
}

template <typename T>
void GArr<T>::assign(const GArr<T> &src)
{
    if (m_totalNum != src.size())
        this->resize(src.size());

    cuSafeCall(cudaMemcpy(m_data, src.begin(), src.size() * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
void GArr<T>::assign(const CArr<T> &src)
{
    if (m_totalNum != src.size())
        this->resize(src.size());

    cuSafeCall(cudaMemcpy(m_data, src.begin(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void GArr<T>::assign(const std::vector<T> &src)
{
    if (m_totalNum != src.size())
        this->resize((uint)src.size());

    cuSafeCall(cudaMemcpy(m_data, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void GArr<T>::assign(const std::vector<T> &src, const uint count, const uint dstOffset, const uint srcOffset)
{
    cuSafeCall(cudaMemcpy(m_data + dstOffset, src.begin() + srcOffset, count * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void GArr<T>::assign(const CArr<T> &src, const uint count, const uint dstOffset, const uint srcOffset)
{
    cuSafeCall(cudaMemcpy(m_data + dstOffset, src.begin() + srcOffset, count * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void GArr<T>::assign(const GArr<T> &src, const uint count, const uint dstOffset, const uint srcOffset)
{
    cuSafeCall(cudaMemcpy(m_data + dstOffset, src.begin() + srcOffset, count * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
void CArr<T>::resize(const uint n)
{
    m_data.resize(n);
}

template <typename T>
void CArr<T>::clear()
{
    m_data.clear();
}

template <typename T>
void CArr<T>::reset()
{
    memset((void *)m_data.data(), 0, m_data.size() * sizeof(T));
}

template <typename T>
void CArr<T>::assign(const GArr<T> &src)
{
    if (m_data.size() != src.size())
        this->resize(src.size());

    cuSafeCall(cudaMemcpy(this->begin(), src.begin(), src.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void CArr<T>::assign(const CArr<T> &src)
{
    if (m_data.size() != src.size())
        this->resize(src.size());

    memcpy(this->begin(), src.begin(), src.size() * sizeof(T));
}

template <typename T>
void CArr<T>::assign(const T &val)
{
    m_data.assign(m_data.size(), val);
}

template <typename T>
void CArr<T>::assign(uint num, const T &val)
{
    m_data.assign(num, val);
}

template <typename T>
void CArr<T>::assign(const std::vector<T> &src, const uint count, const uint dstOffset, const uint srcOffset)
{
    if (m_data.size() != src.size())
        this->resize(src.size());
    memcpy(&m_data[dstOffset], src.begin() + srcOffset, count * sizeof(T));
}

template <typename T>
void CArr<T>::assign(const CArr<T> &src, const uint count, const uint dstOffset, const uint srcOffset)
{
    if (m_data.size() != src.size())
        this->resize(src.size());
    memcpy(&m_data[dstOffset], src.begin() + srcOffset, count * sizeof(T));
}

template <typename T>
void CArr<T>::assign(const GArr<T> &src, const uint count, const uint dstOffset, const uint srcOffset)
{
    if (m_data.size() != src.size())
        this->resize(src.size());
    cuSafeCall(cudaMemcpy(&m_data[dstOffset], src.begin() + srcOffset, count * sizeof(T), cudaMemcpyDeviceToHost));
}

}  // namespace GNNBEM