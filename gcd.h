#ifndef GCD_H
#define GCD_H

#include <inttypes.h>

#include "common.h"

__global__ void findGCDs(uint32_t *nums, int count, char *res, int offset);

#endif