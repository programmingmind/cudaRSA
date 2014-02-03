#define SIZE 100
#define HIGHBIT = 0x80
#define LOWBIT = 0x01

int num[SIZE];

void shiftLeftOne(){
   int flag = 0, flagn = 0;
   for (int i = 0; i < SIZE; i ++) {
      if (num[i] & HIGHBIT) {
         flagn = 1;
      }
      num[i] = num[i] << 1;
      if (flag) {
         num[i] = num[i] + 1;
      }
      flag = flagn;
      flagn = 0;
   }   
}

void shiftRightOne(){
   int flag = 0, flagn = 0;
   for (int i = SIZE - 1; i >= 0; i --) {
      if (num[i] & LOWBIT) {
         flagn = 1;
      }
      num[i] = num[i] >> 1;
      if (flag) {
         num[i] = num[i] | HIGHBIT;
      }
      flag = flagn;
      flagn = 0;
   }   
}

/**
 * Returns true if num1 >= num2
 */
int gteq(int num1[], int num2[]) {
   for(int i = SIZE; i >= 0 ; i--) {
      if (num1[i] != num2[i]) { 
         min = getMin(num1[i], num2[i]);
         return ((num1[i] == min) ? 0 : 1);
      }
   }
   return 1;
}

/**
 * Make sure 1 is > 2
 * result = 1-2
 */
 int * sub(int num1[], int num2[]) {
   int min, result[SIZE];
   for (int i = 0; i < SIZE; i ++) {
      min = getMin(num1[i], num2[i]);
      //normal subtraction
      if (num2[i] == min) {
         result[i] = num1[i] - num2[i]
      } else {
         result[i] = ~(num2[i] - num1[i]) + 1;
         if (num1[i+1] == 0)
            num2[i+1]++; 
         else
            num1[i+1]--;
      }
   }   
 }
 
 function gcd(a, b)
    while a ? b
        if a > b
           a := a - b
        else
           b := b - a
    return a