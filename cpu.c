#include <gmp.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIZE 32 // 1024 bit / 32 bits per int
#define SIZE_BYTES 128 // SIZE * 4 bytes per int
#define HIGHBIT 0x80000000
#define LOWBIT  0x00000001

#define WORK_BYTES 2
#define WORK_SIZE (WORK_BYTES * 8)

#define EQ 0
#define GT 1
#define LT -1

typedef struct bigNum {
   uint32_t num[SIZE];
} bigNum;

void shiftL1(uint32_t num[]);
void shiftR1(uint32_t num[]);
int cmp(uint32_t num1[], uint32_t num2[]);
void Substract(uint32_t num1[], uint32_t num2[]);
void slow_gcd(uint32_t num1[], uint32_t num2[]);
uint32_t* gcd(uint32_t *num1, uint32_t *num2);
void findGCDs(bigNum *nums, int count, char *res, int offset);

uint32_t min(uint32_t a, uint32_t b) {
	return a < b ? a : b;
}

void printCommon(int numKeys, char *res) {
   int countBytes = 1 + ((numKeys - 1) /8);
   int ndx = 0;
   
   for (int i = 0; i < numKeys; i++)
      for (int j = 0; j < countBytes; j++, ndx++)
         if (res[ndx])
            for (int k = 0; k < 8; k++)
               if (res[ndx] & (1 << k))
                  printf("Keys %d %d share a factor\n", i, j*8 + k);
}

//Main function to read keys from file and then matrix, yeah!
int main (int argc, char * argv[]) {
   char *res;
   int countBytes;
   mpz_t tempNum;
   mpz_init(tempNum);
   
   if (argc != 3) {
      printf("error, syntax is %s <file name> <num keys>\n", argv[0]);
      return;
   }
   
   //expects file name as first command line parameter
   FILE *fp = fopen(argv[1], "r");
   //second command line parameter is the numer of keys in the file
   int numKeys = atoi(argv[2]);
   countBytes = 1 + ((numKeys - 1) /8);
   bigNum *numbers = malloc(sizeof(bigNum) * numKeys);
   res = calloc(numKeys, countBytes);
   
   for (int i = 0; i < numKeys; i ++) {
      gmp_fscanf(fp, "%Zd\n", &tempNum);
      mpz_export(numbers + i, NULL, -1, 4, -1, 0, tempNum);
   }
   fclose(fp);
   
   //Lets gcd
   for (int offset = 0;  offset < numKeys; offset += WORK_SIZE) {
   	findGCDs(numbers, numKeys, res, offset);
   }
   printCommon(numKeys, res);
   
   return 0; //!
}

void shiftL1(uint32_t num[]) {
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

void shiftR1(uint32_t num[]) {
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
int cmp(uint32_t num1[], uint32_t num2[]) {
   for (int i = SIZE - 1; i >= 0; i--)
	   if (num1[i] != num2[i])
		   return (num1[i] == min(num1[i], num2[i])) ? LT : GT;
	
	return EQ;
}

// requires that num1 >= num2, num1 -= num2
void Subtract(uint32_t num1[], uint32_t num2[]) {
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
void slow_gcd(uint32_t num1[], uint32_t num2[]) {
   int compare;
	while ((compare = cmp(num1, num2)) != EQ) {
	   if (compare == GT)
		   Subtract(num1, num2);
		else
		   Subtract(num2, num1);
	}
}

// Binary GCD algorithm
// requires num1 > 0 and num2 > 0
// sets either num1 or num2 to the 1 if gcd == 1 or some number >1 if gcd >1 and
// returns the pointer to whichever num was set
uint32_t* gcd(uint32_t *num1, uint32_t *num2) {
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
		Subtract(num2, num1);
	} while (1);
	
	if (shift)
	   shiftL1(num1);
	
	return num1;
}

int greaterOne(uint32_t *num) {
	for (int i = 0; i < SIZE; i++)
		if (i ? num[i] : num[i] > 1)
			return 1;
	return 0;
}

// count is the number of big nums in nums
// res represents a 2 dimensional matrix with at least count bits for each side
// should have count number of threads running, each responsible for 1 row/col
// res will be return as a top diagonal matrix
void findGCDs(bigNum *nums, int count, char *res, int offset) {	
   uint32_t cur[SIZE];
	uint32_t other[SIZE];

	for (int ndx = 0; ndx < count; ndx++) {
		int resOff = ndx * (1 + ((count - 1) / 8));

		// do calc
	   int i = ndx + offset + 1;
	   int limit = min(i + WORK_SIZE, count);
		for (; i < limit; i++) {
			memcpy(cur, nums + ndx, SIZE_BYTES);
			memcpy(other, nums + i, SIZE_BYTES);
			
			if (greaterOne(gcd(cur, other)))
				res[resOff + i / 8] |= 1 << (i % 8);
		}
	}  
}