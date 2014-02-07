// driver for rsa cracking

#include <gmp.h>

#include "gcd.h"

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
   char *res, *cudaRes;
   uint32_t *numbers, *cudaNums;
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
   countBytes = 1 + ((numKeys - 1) /8);
   int numSize = SIZE * sizeof(uint32_t) * numKeys;

   numbers = (uint32_t *) malloc(numSize);
   cudaMalloc(&cudaNums, numSize);

   res = (char *) calloc(numKeys, countBytes);
   cudaMalloc(&cudaRes, numKeys * countBytes);
   cudaMemset(cudaRes, 0, numKeys * countBytes);
   
   for (int i = 0; i < numKeys; i ++) {
      gmp_fscanf(fp, "%Zd\n", &tempNum);
      mpz_export(numbers + i, NULL, -1, 4, -1, 0, tempNum);
   }
   fclose(fp);

   cudaMemcpy(cudaNums, numbers, numSize, cudaMemcpyHostToDevice);
   
   //Lets gcd
   for (int offset = 0;  offset < numKeys; offset += WORK_SIZE) {
      // <<<dimGrid, dimBlock>>>
      findGCDs<<<1 + ((numKeys - 1) / SIZE), SIZE>>>(numbers, numKeys, res, offset);
   }

   cudaMemcpy(res, cudaRes, numKeys * countBytes, cudaMemcpyDeviceToHost);

   cudaFree(cudaNums);
   cudaFree(cudaRes);
   
   printCommon(numKeys, res);
   
   return 0; //!
}
