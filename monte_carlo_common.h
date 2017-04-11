

#pragma once

#include <vector>


double diff_ts_us(struct timespec t1, struct timespec t2)
{
    return ((((t1.tv_sec - t2.tv_sec) * 1e9) + 
                t1.tv_nsec - t2.tv_nsec))/1000;
}