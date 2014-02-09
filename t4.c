#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <gmp.h>

static int seedOff = 0;

void generateRandomPrime(mpz_t *num);

void encrypt(mpz_t *ciphertext, mpz_t m, mpz_t e, mpz_t n);
void decrypt(mpz_t *message, mpz_t c, mpz_t d, mpz_t n);

int main (int argc, char *argv[]) {
   mpz_t c, m1, m2, pc, q1, q2, e, d1, d2, phi1, phi2, tmp1, tmp2,
    key1, key2;
   FILE *fp, *fp1;
   char chars[3];

   //initializations
   mpz_inits(c, m1, m2, pc, q1, q2, e, d1, d2, phi1, phi2, tmp1, tmp2,
    key1, key2, NULL);

   //Welcome!
   printf("Begin task 4 / 5...\n");
   printf("*******************************\n");
   printf("Welcome to fun with big numbers\n");
   printf("Lets do some decryption\n");
   printf("*******************************\n");

   //Get needed input from file
   fp = fopen("p4in.txt", "r");
   gmp_fscanf(fp, "%Zd\n%Zd\n%Zd\n%Zd", &key1, &key2, &pc, &c);
   gmp_printf("key1=%Zd\n\nkey2=%Zd\n\nCommon factor=%Zd\n\ncipher=%Zd\n\n",
    key1, key2, pc, c);
   fclose(fp);

   //setup calculating needed values
   mpz_cdiv_q(q1, key1, pc);
   mpz_cdiv_q(q2, key2, pc);
   mpz_set_ui(e, 65537);
   mpz_sub_ui(tmp1, pc, 1);
   mpz_sub_ui(tmp2, q1, 1);
   mpz_mul(phi1, tmp1, tmp2);
   mpz_sub_ui(tmp2, q2, 1);
   mpz_mul(phi2, tmp1, tmp2);
   mpz_invert(d1, e, phi1);
   mpz_invert(d2, e, phi2);

   //Lets try and decrypt
   decrypt(&m1, c, d1, key1);
   decrypt(&m2, c, d2, key2);

   fp = fopen("p4out.txt", "w");
   gmp_fprintf(fp, "%Zx\n%Zx\n", m1, m2);
   fclose(fp);
   mpz_clears(c, m1, m2, pc, q1, q2, e, d1, d2, phi1, phi2, tmp1, tmp2,
    key1, key2, NULL);

   fp = fopen("p4out.txt", "r");
   chars[2] = 0;
   while (fscanf(fp, "%c%c", chars, chars + 1) > 0){
     printf("%c", (int)strtol(chars, NULL, 16));
   }
   fclose(fp);

   return 0;
}

void encrypt(mpz_t *ciphertext, mpz_t m, mpz_t e, mpz_t n){
   mpz_powm(*ciphertext, m, e, n);
}

void decrypt(mpz_t *message, mpz_t c, mpz_t d, mpz_t n){
   mpz_powm(*message, c, d, n);
}

void generateRandomPrime(mpz_t *num) {
   mpz_t temp;
   mp_bitcnt_t bits = 512;
   gmp_randstate_t state;

   seedOff++; 
   mpz_init(temp);

   time_t result = time(NULL);
   gmp_randinit_mt(state);
   gmp_randseed_ui(state, result + seedOff);

   mpz_urandomb(temp, state, bits);

   mpz_nextprime(*num, temp);
}
