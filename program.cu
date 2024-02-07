#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "file.hpp"
#include "holders.hpp"

int main(int __argc,char *__argv[]){
    if(__argc != 2){
        printf("there must be 1 argument");
        return -1;
    }
    neuralnetwork NN = FromFile(__argv[0]);
    int NumofFirst = NN.layers[0].NumOfNu;
    int NumofLast = NN.layers[NN.NumOfLayers-1].NumOfNu;
    neuralnetwork* Nptr;
    cudaMallocManaged((void**)&Nptr,sizeof(neuralnetwork));
    Nptr = &NN;
    

    cudaFree(Nptr);
    return 0;
}