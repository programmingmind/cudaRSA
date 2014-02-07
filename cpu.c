#include <inttypes.h>

#define SIZE 32 // 1024 bit / 32 bits per int
#define SIZE_BYTES 128 // SIZE * 4 bytes per int
#define HIGHBIT 0x80000000
#define LOWBIT  0x00000001

#define EQ 0
#define GT 1
#define LT -1

void shiftL1(uint32_t num[]);
void shiftR1(uint32_t num[]);
int cmp(uint32_t num1[], uint32_t num2[]);
void Substract(uint32_t num1[], uint32_t num2[]);
void slow_gcd(uint32_t num1[], uint32_t num2[]);
uint32_t* gcd(uint32_t *num1, uint32_t *num2);
void findGCDs(uint32_t *nums, int count, char *res);

typedef struct bigNum {
   uint32_t num[SIZE];
} bigNum;

void printCommon(int numKeys, char *res) {
   int countBytes = (1 + ((numKeys - 1) /8);
   int ndx = 0;
   
   for (int i = 0; i < numKeys; i++)
      for (int j = 0; j < countBytes; j++, ndx++) {
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
      printf("error, syntax is %s <file name> <num keys>", argv[0]);
   }
   
   //expects file name as first command line parameter
   FILE *fp = fopen(argv[1], "r");
   //second command line parameter is the numer of keys in the file
   int numKeys = atoi(argv[2]);
   countBytes = (1 + ((numKeys - 1) /8);
   bigNum *numbers = malloc(sizeof(bigNum) * numKeys);
   res = calloc(numKeys, countBytes);
   
   for (int i = 0; i < numKeys; i ++) {
      gmp_fscanf(fp, "%Zd\n", &tempNum);
      mpz_export(numbers + i, NULL, -1, 4, -1, 0, tempNum);
   }
   fclose(fp);
   
   //Lets gcd
   findGCDs(numbers, numKeys, res);
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
   for (int i = 0; i < SIZE: i++) {
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
		   cuSubtract(num1, num2);
		else
		   cuSubtract(num2, num1);
	}
}

// Binary GCD algorithm
// requires num1 > 0 and num2 > 0
// sets either num1 or num2 to the gcd and returns the pointer to that num
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
		cuSubtract(num2, num1);
	} while (1);
	
	while (shift--)
	   shiftL1(num1);
	
	return num1;
}

// count is the number of big nums in nums
// res represents a 2 dimensional matrix with at least count bits for each side
// should have count number of threads running, each responsible for 1 row/col
// res will be return as a top diagonal matrix
void findGCDs(uint32_t *nums, int count, char *res) {
   uint32_t ONE[SIZE];
	
	int countBytes = 1 + ((count - 1) / 8);
	
	memset(ONE, 0, SIZE_BYTES);
   ONE[0] = 1;
	
	char *row;
   row = malloc(countBytes);
	
	uint32_t cur[SIZE];
	uint32_t other[SIZE];
	
   // do calc
   for (int ndx = 0; ndx < count; ndx++) {
      memset(row, 0, countBytes);
      for (int i = ndx + 1; i < count; i++) {
         memcpy(nums + ndx * SIZE_BYTES, cur, SIZE_BYTES);
         memcpy(nums + i * SIZE_BYTES, other, SIZE_BYTES);
         
         if (cmp(gcd(cur, other), ONE) == GT)
            row[ndx / 8] |= 1 << (ndx % 8);
      }
      // write row
      memcpy(res + ndx*countBytes, row, countBytes);
   }
   
	free(row);
}
