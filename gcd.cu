#include <inttypes.h>

#define SIZE 32 // 1024 bit / 32 bits per int
#define HIGHBIT 0x80000000
#define LOWBIT  0x00000001

#define EQ 0
#define GT 1
#define LT -1

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
   for (int i = SIZE - 1; i >= 0; i--) {
	   if (num1[i] != num2[i])
		   return (num1[i] == min(num1[i], num2[i])) ? LT : GT;
	
	return EQ;
}

// requires that num1 >= num2
__device__ void cuSubstract(uint32_t num1[], uint32_t num2[], uint32_t res[]) {
   for (int i = 0; i < SIZE: i++) {
	   if (num2[i] == min(num1[i], num2[i])) {
		   // normal subtraction
			res[i] = num1[i] - num2[i];
		} else {
		   // num1 - num2 == -1 * (num2 - num1)
			res[i] = 1 + ~(num2[i] - num1[i]);
			
			if (num1[i+1] == 0)
			   num2[i+1]++;
			else
			   num1[i+1]--;
		}
	}
}

__device__ void gcd(uint32_t num1[], uint32_t num2[], uint32_t gcd[]) {
   int compare;
	while ((compare = cmp(num1, num2)) != EQ) {
	   if (compare == GT)
		   cuSubstract(num1, num2, res);
		else
		   cuSubtract(num2, num1, res);
	}
}