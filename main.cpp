
#include <iostream>
#include <thread>
#include <cstdlib>
#include <cstring>
#include <getopt.h>

#include <time.h>

#include "monte_carlo_cpu.h"

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
    if (argc == 1) {
        std::cerr << ::basename(argv[0]) << ": missing option(s)\n" << std::endl;
        ::usage(argv[0]);
        return 1;
    }

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

    std::string config_path(argv[1]);

    MonteCarloCPU monte_carlo;
    monte_carlo.calculate();

    return 0;
}
