#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "file.h"


#define N 10000
#define S sizeof(float)


__global__ void kernel(){


}



int main(int __argc,char *__argv[]){
    if(__argc != 2){
        printf("there must be 1 argument");
        return -1;
    }
    




    return 0;
}