

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#include "monte_carlo_common.h"

class MonteCarloCPU
{
private:
	std::random_device m_rd;
    std::mt19937 m_gen;
    std::normal_distribution<> m_dist;

public:
	MonteCarloCPU()
		: m_rd(),
		  m_gen(m_rd()),
		  m_dist(70,10)
	{
	}

private:
	double call_price(const int& num_sims, const double& S, const double& K, const double& r, const double& v, const double& T) 
	{
		double S_adjust = S * exp(T*(r-0.5*v*v));
		double S_cur = 0.0;
		double payoff_sum = 0.0;

		for (int i=0; i<num_sims; i++) {
			double rand_num = m_dist(m_gen);
			S_cur = S_adjust * exp(sqrt(v*v*T)*rand_num);
			payoff_sum += std::max(S_cur - K, 0.0);
		}

		return (payoff_sum / static_cast<double>(num_sims)) * exp(-r*T);
	}

	double put_price(const int& num_sims, const double& S, const double& K, const double& r, const double& v, const double& T) 
	{
		double S_adjust = S * exp(T*(r-0.5*v*v));
		double S_cur = 0.0;
		double payoff_sum = 0.0;

		for (int i=0; i<num_sims; i++) {
			double rand_num = m_dist(m_gen);
			S_cur = S_adjust * exp(sqrt(v*v*T)*rand_num);
			payoff_sum += std::max(K - S_cur, 0.0);
		}

		return (payoff_sum / static_cast<double>(num_sims)) * exp(-r*T);
	}

public:
	void calculate()
	{
	// First we create the parameter list
    int num_sims = 10000000;   // Number of simulated asset paths
    double S = 143.34;  // Option price
    double K = 119.0;  // Strike price
    double r = 0.10304;   // Risk-free rate (5%)
    double v = 0.48;    // Volatility of the underlying (20%)
    double T = 1/12;    // One year until expiry

    struct timespec start_ts, end_ts;

    clock_gettime(CLOCK_MONOTONIC, &start_ts);  

    // Then we calculate the call/put values via Monte Carlo
    double call = call_price(num_sims, S, K, r, v, T);
    double put = put_price(num_sims, S, K, r, v, T);

    clock_gettime(CLOCK_MONOTONIC, &end_ts);


    // Finally we output the parameters and prices
    std::cout << "Number of Paths:      " << num_sims << std::endl;
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
