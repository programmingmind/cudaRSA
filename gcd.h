#ifndef GCD_H
#define GCD_H

#include <inttypes.h>

#include "common.h"

__global__ void findGCDs(bigInt *nums, int count, char *res, int offset);

#endif