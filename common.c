#include <gmp.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

void printCommon(int numKeys, char *res) {
   int i,j,k;

   int countBytes = 1 + ((numKeys - 1) /8);
   int ndx = 0;
   
   for (i = 0; i < numKeys; i++)
      for (j = 0; j < countBytes; j++, ndx++)
         if (res[ndx])
            for (k = 0; k < 8; k++)
               if (res[ndx] & (1 << k))
                  printf("Keys %d %d share a factor\n", i, j*8 + k);
}

int readFile(const char *fileName, uint32_t **numbers, char **res) {
   int countBytes;
   mpz_t tempNum;
   mpz_init(tempNum);
      
  
   FILE *fp = fopen(fileName, "r");
   
   int numKeys = 0;
   while (EOF != (fscanf(fp, "%*[^\n]"), fscanf(fp, "%*c"))) 
      ++numKeys;

   printf("%d keys\n", numKeys);

   fseek(fp, 0, SEEK_SET);

   countBytes = 1 + ((numKeys - 1) /8);
   *numbers = (uint32_t *) malloc(SIZE_BYTES * numKeys);
   *res = (char *) calloc(numKeys, countBytes);
   
   int i;
   for (i = 0; i < numKeys; i ++) {
      gmp_fscanf(fp, "%Zd\n", &tempNum);
      mpz_export((*numbers) + i * SIZE, NULL, -1, 4, -1, 0, tempNum);
   }
   fclose(fp);
   
   return numKeys;
}