

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "monte_carlo_common.h"

class MonteCarloCPU
{
    std::vector<float> m_normals;

public:
    explicit MonteCarloCPU(const std::vector<float>& _normals)
            : m_normals(_normals)
    {
    }

private:
    double call_price(const double S,
                      const double K,
                      const double r,
                      const double v,
                      const double T,
                      const size_t num_steps,
                      const size_t num_paths)
    {
        double S_adjust = S * exp(T*(r-0.5*v*v));
        double S_cur = 0.0;
        double payoff_sum = 0.0;

        for (int i = 0; i < (num_steps * num_paths); i++) {
            S_cur = S_adjust * exp(sqrt(v*v*T)*m_normals[i]);
            payoff_sum += std::max(S_cur - K, 0.0);
        }

        return (payoff_sum / static_cast<double>(num_steps * num_paths)) * exp(-r*T);
    }

    double put_price(const double S,
                     const double K,
                     const double r,
                     const double v,
                     const double T,
                     const size_t num_steps,
                     const size_t num_paths)
    {
        double S_adjust = S * exp(T*(r-0.5*v*v));
        double S_cur = 0.0;
        double payoff_sum = 0.0;

        for (int i = 0; i < (num_steps * num_paths); i++) {
            S_cur = S_adjust * exp(sqrt(v*v*T)*m_normals[i]);
            payoff_sum += std::max(K - S_cur, 0.0);
        }

        return (payoff_sum / static_cast<double>(num_steps * num_paths)) * exp(-r*T);
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
        struct timespec start_ts, end_ts;

        clock_gettime(CLOCK_MONOTONIC, &start_ts);

        // Then we calculate the call/put values via Monte Carlo
        double call = call_price(S, K, r, v, T, num_steps, num_paths);
        double put = put_price(S, K, r, v, T, num_steps, num_paths);

        clock_gettime(CLOCK_MONOTONIC, &end_ts);


        std::cout << "==== CPU Results =====" << std::endl;
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
