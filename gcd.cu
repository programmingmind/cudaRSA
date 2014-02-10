#include "gcd.h"

__device__ void shiftL1(bigInt num[]) {
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

__device__ void shiftR1(bigInt num[]) {
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
__device__ int cmp(bigInt num1[], bigInt num2[]) {
   for (int i = SIZE - 1; i >= 0; i--)
	   if (num1[i] != num2[i])
		   return (num1[i] < num2[i]) ? LT : GT;
	
	return EQ;
}

// requires that num1 >= num2, num1 -= num2
__device__ void cuSubtract(bigInt num1[], bigInt num2[]) {
   for (int i = 0; i < SIZE; i++) {
	   if (num2[i] <= num1[i]) {
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
__device__ void slow_gcd(bigInt num1[], bigInt num2[]) {
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
__device__ bigInt* gcd(bigInt *num1, bigInt *num2) {
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
		   bigInt *t = num1;
			num1 = num2;
			num2 = t;
		}
		cuSubtract(num2, num1);
	} while (1);
	
	if (shift)
	   shiftL1(num1);
	
	return num1;
}

__device__ bool greaterOne(bigInt *num) {
	for (int i = 0; i < SIZE; i++)
		if (i ? num[i] : num[i] > 1)
			return true;
	return false;
}

// count is the number of big nums in nums
// res represents a 2 dimensional matrix with at least count bits for each side
// should have count number of threads running, each responsible for 1 row/col
// res will be return as a top diagonal matrix
__global__ void findGCDs(bigInt *nums, int count, char *res, int offset) {	
   int ndx = blockIdx.x * blockDim.x + threadIdx.x; // == offset in bits
	int resOff = ndx * (1 + ((count - 1) / 8));

	bigInt cur[SIZE];
	bigInt other[SIZE];
	
   // do calc
   int i = ndx + offset + 1;
   int limit = min(i + WORK_SIZE, count);
	for (; i < limit; i++) {
		memcpy(cur, nums + ndx * SIZE, SIZE_BYTES);
		memcpy(other, nums + i * SIZE, SIZE_BYTES);
		
		if (greaterOne(gcd(cur, other)))
			res[resOff + i / 8] |= 1 << (i % 8);
	}
}
