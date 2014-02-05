#ifndef GCD_H
#define GCD_H

#include <inttypes.h>

#define SIZE 32 // 1024 bit / 32 bits per int
#define SIZE_BYTES 128 // SIZE * 4 bytes per int
#define HIGHBIT 0x80000000
#define LOWBIT  0x00000001

#define EQ 0
#define GT 1
#define LT -1

__global__ void findGCDs(uint32_t *nums, int count, char *res);

#endif