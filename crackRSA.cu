// driver for rsa cracking

#include <stdio.h>

#include "common.h"
#include "gcd.h"

//Main function to read keys from file and then matrix, yeah!
int main (int argc, char * argv[]) {
   char *res, *cudaRes;
   bigInt *numbers, *cudaNums;
   
   if (argc != 2) {
      printf("error, syntax is %s <file name>\n", argv[0]);
      return 0;
   }

   int numKeys = readFile(argv[1], &numbers, &res);
   int countBytes = 1 + ((numKeys - 1) /8);
   int numSize = SIZE_BYTES * numKeys;

   cudaMalloc(&cudaNums, numSize);
   cudaMalloc(&cudaRes, numKeys * countBytes);
   cudaMemset(cudaRes, 0, numKeys * countBytes);
   
   cudaMemcpy(cudaNums, numbers, numSize, cudaMemcpyHostToDevice);
   
   int dimBlock = SIZE*2;
   int dimGrid = 1 + ((numKeys - 1) / dimBlock);

   printf("%d blocks of size %d\n", dimGrid, dimBlock);

   //Lets gcd
   for (int offset = 0;  offset < numKeys; offset += WORK_SIZE) {
      // <<<dimGrid, dimBlock>>>
      findGCDs<<<dimGrid, dimBlock>>>(cudaNums, numKeys, cudaRes, offset);
   }

   cudaMemcpy(res, cudaRes, numKeys * countBytes, cudaMemcpyDeviceToHost);

   cudaFree(cudaNums);
   cudaFree(cudaRes);
   
   writeFiles("privateKeys", numKeys, numbers, res);

   free(numbers);
   free(res);
   
   return 0;
}
