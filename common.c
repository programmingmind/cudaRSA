#include <gmp.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

int readFile(const char *fileName, bigInt **numbers, char **res) {
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
   *numbers = (bigInt *) malloc(SIZE_BYTES * numKeys);
   *res = (char *) calloc(numKeys, countBytes);
   
   int i;
   for (i = 0; i < numKeys; i ++) {
      gmp_fscanf(fp, "%Zd\n", &tempNum);
      mpz_export((*numbers) + i * SIZE, NULL, -1, BIGINT_SIZE, -1, 0, tempNum);
   }
   fclose(fp);
   
   return numKeys;
}

void computePrivate(mpz_t pb1, mpz_t pb2, mpz_t *pk1, mpz_t *pk2) {
   mpz_t common, q1, q2, e, t1, t2, tc, tot1, tot2;
   mpz_inits(common, q1, q2, e, t1, t2, tc, tot1, tot2, NULL);

   mpz_gcd(common, pb1, pb2);
   mpz_cdiv_q(q1, pb1, common);
   mpz_cdiv_q(q2, pb2, common);

   mpz_sub_ui(t1, q1, 1);
   mpz_sub_ui(t2, q2, 1);
   mpz_sub_ui(tc, common, 1);

   mpz_mul(tot1, t1, tc);
   mpz_mul(tot2, t2, tc);

   mpz_set_ui(e, 65537);

   mpz_invert(*pk1, e, tot1);
   mpz_invert(*pk2, e, tot2);
}

void writeFiles(const char *privateFile,int numKeys,bigInt *keys,char *res) {

   int i,j,k;
   mpz_t k1, k2, pk1, pk2;

   mpz_inits(k1, k2, pk1, pk2, NULL);

   int countBytes = 1 + ((numKeys - 1) /8);
   int ndx = 0;

   FILE *priv = fopen(privateFile, "w");
   
   for (i = 0; i < numKeys; i++) {
      for (j = 0; j < countBytes; j++, ndx++) {
         if (res[ndx]) {
            for (k = 0; k < 8; k++) {
               if (res[ndx] & (1 << k)) {
                  printf("Keys %d %d share a factor\n", i, j*8 + k);

                  mpz_import(k1, SIZE, -1, BIGINT_SIZE, -1, 0, keys + i * SIZE);
                  mpz_import(k2, SIZE, -1, BIGINT_SIZE, -1, 0, keys + (j*8 + k) * SIZE);
                  computePrivate(k1, k2, &pk1, &pk2);

                  gmp_fprintf(priv, "%Zd:%Zd\n%Zd:%Zd\n", k1, pk1, k2, pk2);
               }
            }
         }
      }
   }

   fclose(priv);
}