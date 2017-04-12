#pragma once 

#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

template <class T>
class DeviceArray
{
public:
    DeviceArray()
        : m_start(0),
          m_end(0)
    {}

    explicit DeviceArray(size_t size)
    {
        allocate(size);
    }

    ~DeviceArray()
    {
        free();
    }

    void resize(size_t size)
    {
        free();
        allocate(size);
    }

    size_t get_size() const
    {
        return m_end - m_start;
    }

    const T* get_data() const
    {
        return m_start;
    }

    T* get_data()
    {
        return m_start;
    }

    void set(const T* src, size_t size)
    {
        size_t min = std::min(size, get_size());
        cudaError_t result = cudaMemcpy(m_start, src, min * sizeof(T), cudaMemcpyHostToDevice);
        if (result != cudaSuccess)
        {
            throw std::runtime_error("failed to copy to device memory");
        }
    }

    void get(T* dest, size_t size)
    {
        size_t min = std::min(size, get_size());
        cudaError_t result = cudaMemcpy(dest, m_start, min * sizeof(T), cudaMemcpyDeviceToHost);
        if (result != cudaSuccess)
        {
            throw std::runtime_error("failed to copy to host memory");
        }
    }


private:
    void allocate(size_t size)
    {
        cudaError_t result = cudaMalloc((void**)&m_start, size * sizeof(T));
        if (result != cudaSuccess)
        {
            m_start = m_end = 0;
            char tmp[100];
            snprintf(tmp, sizeof(tmp), "failed to allocate(%zu) device memory", size);
            throw std::runtime_error(tmp);
        }
        m_end = m_start + size;
    }

    void free()
    {
        if (m_start != 0)
        {
            cudaFree(m_start);
            m_start = m_end = 0;
        }
    }

private:
    T* m_start;
    T* m_end;
};

