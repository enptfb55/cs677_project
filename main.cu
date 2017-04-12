
#include <iostream>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <getopt.h>

#include <time.h>

#include <curand.h>

#include "monte_carlo_cpu.h"
#include "monte_carlo_gpu.h"

void usage(const char* app)
{
    std::cout << "Usage: " << ::basename(app) << " [-h] config_path" << std::endl;
    std::cout << "Options:\n"
              << "  -h, --help              This help message\n"
              << "  config_path             Path to config file\n"
              << std::endl;
}

int main(int argc, char* argv[])
{
    /*
    if (argc == 1) {
        std::cerr << ::basename(argv[0]) << ": missing option(s)\n" << std::endl;
        ::usage(argv[0]);
        return 1;
    }
    */

    while (true) {
        static option long_options[] = {
            {"help", no_argument, NULL, 'h'},
            {0, 0, 0, 0},
        };

        const int c = ::getopt_long(argc, argv, "h", long_options, nullptr);
        if (c == -1) break;

        switch (c) {
            case 'h':
            case '?':
            default:
            ::usage(argv[0]);
            return 1;
        }
    } 

    //std::string config_path(argv[1]);

    size_t num_paths = 10000000;   // Number of simulated asset paths
    size_t num_steps = 1;
    size_t num_normals = num_paths * num_steps;
    double S = 100.0;  // Option price
    double K = 100.0;  // Strike price
    double r = 0.05;   // Risk-free rate (5%)
    double v = 0.2;    // Volatility of the underlying (20%)
    double T = 1;    // One year until expiry
    float dt = float(T)/float(num_steps);
    float sqrdt = sqrt(dt);

    DeviceArray<float> d_normals(num_normals);

    curandGenerator_t curandGenerator;
    curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL) ;
    curandGenerateNormal(curandGenerator, d_normals.get_data(), num_normals, 0.0f, sqrdt);

    std::vector<float> h_normals(num_normals);
    d_normals.get(&h_normals[0], num_normals);

    MonteCarloCPU monte_carlo_cpu(h_normals);
    monte_carlo_cpu.calculate(num_normals, S, K, r, v, T);

    MonteCarloGPU monte_carlo_gpu(&d_normals);
    monte_carlo_gpu.calculate(num_steps, num_paths, S, K, r, v, T);

    return 0;
}
