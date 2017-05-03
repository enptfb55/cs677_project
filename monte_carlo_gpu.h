
#pragma once

#include "monte_carlo_common.h"

#include "device_array.h"

__global__ void reduce_price(float *d_price_array, unsigned int n)
{
    extern __shared__ float sdata[];

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? d_price_array[i] : 0;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0)
        atomicAdd(&d_price_array[0], sdata[0]);

}


__global__ void call_price(float *d_output, 
                           float *d_normals, 
                           double S,
                           double K,
                           double r,
                           double v,
                           double T,
                           size_t num_steps,
                           size_t num_paths)
{
    int n_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int s_idx = n_idx;

    double S_adjust = S * exp(T*(r-0.5*v*v));
    double S_cur = 0.0;
    double payoff_sum = 0.0;

    if (n_idx < num_paths) {
        int n = 0;
        do {
            S_cur = S_adjust * exp(sqrt(v*v*T)*d_normals[n_idx]);
            n_idx++;
            n++;
        }
        while (n < num_steps);
        payoff_sum = (S_cur > K ? S_cur - K : 0.0);
        __syncthreads();
        d_output[s_idx] = exp(-r*T) * payoff_sum;
    }
}

__global__ void put_price(float *d_output, 
                           float *d_normals, 
                           double S,
                           double K,
                           double r,
                           double v,
                           double T,
                           size_t num_steps,
                           size_t num_paths)
{
    int n_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int s_idx = n_idx;

    double S_adjust = S * exp(T*(r-0.5*v*v));
    double S_cur = 0.0;
    double payoff_sum = 0.0;

    if (n_idx < num_paths) {
        int n = 0;
        do {
            S_cur = S_adjust * exp(sqrt(v*v*T)*d_normals[n_idx]);
            n_idx++;
            n++;
        }
        while (n < num_steps);
        payoff_sum = (S_cur > K ? 0.0 : K - S_cur);
        __syncthreads();
        d_output[s_idx] = exp(-r*T) * payoff_sum;
    }
}

class MonteCarloGPU
{
    DeviceArray<float> *m_normals;

public:
    explicit MonteCarloGPU(DeviceArray<float>* _normals)
            : m_normals(_normals)
    {
    }

public:
    void calculate(size_t num_steps, 
                   size_t num_paths, 
                   double S, 
                   double K, 
                   double r, 
                   double v, 
                   double T)
    {
        const size_t block_size = 1024;
        const size_t grid_size = ceil(float(num_paths)/float(block_size));
        const size_t shm_size = (size_t)block_size * sizeof(int);

        std::cout << "block_size=" << block_size << std::endl;
        std::cout << "grid_size="  << grid_size << std::endl;
        std::cout << "shm_size=" << shm_size << std::endl;

        struct timespec start_ts, end_ts;

        DeviceArray<float> d_call_output(num_paths);
        DeviceArray<float> d_put_output(num_paths);

        clock_gettime(CLOCK_MONOTONIC, &start_ts);

        call_price<<<grid_size, block_size>>>(d_call_output.get_data(), m_normals->get_data(), S, K, r, v, T, num_steps, num_paths);
        put_price<<<grid_size, block_size>>>(d_put_output.get_data(), m_normals->get_data(), S, K, r, v, T, num_steps, num_paths);

        reduce_price<<<grid_size, block_size, shm_size>>>(d_call_output.get_data(), num_paths);
        reduce_price<<<grid_size, block_size, shm_size>>>(d_put_output.get_data(), num_paths);

        cudaDeviceSynchronize();

        std::vector<float> h_call_output(1);
        std::vector<float> h_put_output(1);

        d_call_output.get(&h_call_output[0], 1);
        d_put_output.get(&h_put_output[0], 1);

        double call = h_call_output[0];
        call/=num_paths;

        double put = 0.0;
        put = h_put_output[0];
        put /= num_paths;

        clock_gettime(CLOCK_MONOTONIC, &end_ts);

        std::cout << "==== GPU Results =====" << std::endl;
        std::cout << "Number of Paths:      " << num_paths << std::endl;
        std::cout << "Number of Steps:      " << num_steps << std::endl;
        std::cout << "Underlying:           " << S << std::endl;
        std::cout << "Strike:               " << K << std::endl;
        std::cout << "Risk-Free Rate:       " << r << std::endl;
        std::cout << "Volatility:           " << v << std::endl;
        std::cout << "Maturity:             " << T << std::endl;

        std::cout << "Call Price:           " << call << std::endl;
        std::cout << "Put Price:            " << put << std::endl;

        std::cout << "Time Elapsed(ms):     " << diff_ts_us(end_ts, start_ts) / 1000 << std::endl;
    }

};

