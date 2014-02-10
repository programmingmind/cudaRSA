#ifndef COMMON_H
#define COMMON_H

#include <gmp.h>
#include <inttypes.h>

#define SIZE 32 // 1024 bit / 32 bits per int
#define SIZE_BYTES 128 // SIZE * 4 bytes per int
#define HIGHBIT 0x80000000
#define LOWBIT  0x00000001

#define WORK_BYTES 2
#define WORK_SIZE (WORK_BYTES * 8)

#define EQ 0
#define GT 1
#define LT -1

int readFile(const char *fileName, uint32_t **numbers, char **res);

void computePrivate(mpz_t pb1, mpz_t pb2, mpz_t *pk1, mpz_t *pk2);

void writeFiles(const char *publicFile, const char *privateFile, int numKeys,
 uint32_t *keys, char *res);

#endif