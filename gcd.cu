#include "gcd.h"

__device__ void shiftL1(uint32_t num[]) {
   int flag = 0, flagn = 0;
	for (int i = 0; i < SIZE; i++) {
	   if (num[i] & HIGHBIT)
		   flagn = 1;
		
		num[i] <<= 1;
		if (flag)
		   num[i]++;
		
		flag = flagn;
		flagn = 0;
	}
}

__device__ void shiftR1(uint32_t num[]) {
   int flag = 0, flagn = 0;
	for (int i = SIZE - 1; i >= 0; i--) {
	   if (num[i] & LOWBIT)
		   flagn = 1;
		
		num[i] >>= 1;
		if (flag)
		   num[i] |= HIGHBIT;
		
		flag = flagn;
		flagn = 0;
	}
}

// returns num1 (LT,EQ,GT)? num2
__device__ int cmp(uint32_t num1[], uint32_t num2[]) {
   for (int i = SIZE - 1; i >= 0; i--)
	   if (num1[i] != num2[i])
		   return (num1[i] == min(num1[i], num2[i])) ? LT : GT;
	
	return EQ;
}

// requires that num1 >= num2, num1 -= num2
__device__ void cuSubtract(uint32_t num1[], uint32_t num2[]) {
   for (int i = 0; i < SIZE; i++) {
	   if (num2[i] == min(num1[i], num2[i])) {
		   // normal subtraction
			num1[i] = num1[i] - num2[i];
		} else {
		   // num1 - num2 == -1 * (num2 - num1)
			num1[i] = 1 + ~(num2[i] - num1[i]);
			
			if (num1[i+1] == 0)
			   num2[i+1]++;
			else
			   num1[i+1]--;
		}
	}
}

// eulers gcd algorithm without modulus
__device__ void slow_gcd(uint32_t num1[], uint32_t num2[]) {
   int compare;
	while ((compare = cmp(num1, num2)) != EQ) {
	   if (compare == GT)
		   cuSubtract(num1, num2);
		else
		   cuSubtract(num2, num1);
	}
}

// Binary GCD algorithm
// requires num1 > 0 and num2 > 0
// sets either num1 or num2 to the 1 if gcd == 1 or some number >1 if gcd >1 and
// returns the pointer to whichever num was set
__device__ uint32_t* gcd(uint32_t *num1, uint32_t *num2) {
   int shift, compare;
	
	for (shift = 0; ((num1[0] | num2[0]) & LOWBIT) == 0; ++shift) {
	   shiftR1(num1);
		shiftR1(num2);
	}
	
	while ((num1[0] & 1) == 0)
	   shiftR1(num1);
	
	do {
	   while ((num2[0] & 1) == 0)
		   shiftR1(num2);
		
		compare = cmp(num1, num2);
		if (compare == EQ)
		   break;
		else if (compare == GT) {
		   uint32_t *t = num1;
			num1 = num2;
			num2 = t;
		}
		cuSubtract(num2, num1);
	} while (1);
	
	if (shift)
	   shiftL1(num1);
	
	return num1;
}

// count is the number of big nums in nums
// res represents a 2 dimensional matrix with at least count bits for each side
// should have count number of threads running, each responsible for 1 row/col
// res will be return as a top diagonal matrix
__global__ void findGCDs(uint32_t *nums, int count, char *res) {
   __shared__ uint32_t ONE[SIZE];
	
   int ndx = blockIdx.x * blockDim.x + threadIdx.x; // == offset in bits
	int countBytes = 1 + ((count - 1) / 8);
	
	if (threadIdx.x == 0) {
	    memset(ONE, 0, SIZE_BYTES);
		 ONE[0] = 1;
	}
	__syncthreads();
	
	char *row;
   row = (char *) malloc(countBytes);
	memset(row, 0, countBytes);
	
	uint32_t cur[SIZE];
	uint32_t other[SIZE];
	
   // do calc
	for (int i = ndx + 1; i < count; i++) {
		memcpy(nums + ndx * SIZE_BYTES, cur, SIZE_BYTES);
		memcpy(nums + i * SIZE_BYTES, other, SIZE_BYTES);
		
		if (cmp(gcd(cur, other), ONE) == GT)
		   row[ndx / 8] |= 1 << (ndx % 8);
	}
	
	// write row
	memcpy(res + ndx*countBytes, row, countBytes);
	
	free(row);
}
