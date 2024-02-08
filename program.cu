#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "file.hpp"
#include "holders.hpp"
#include "mnist.hpp"
#include "calc.cu"

int main(int __argc,char *__argv[]){
    if(__argc != 5){
        printf("there must be 5 arguments");
        return -1;
    }
    neuralnetwork NN = FromFile(__argv[0]);
    int NumofFirst = NN.layers[0].NumOfNu;
    int NumofLast = NN.layers[NN.NumOfLayers-1].NumOfNu;
    neuralnetwork* Nptr;
    cudaMallocManaged((void**)&Nptr,sizeof(neuralnetwork));
    Nptr = &NN;
    byte** ITNN = InputsToNN(__argv[1],(int)__argv[3]);
    byte** EFNN = ExpectedFromNN(__argv[2],(int)__argv[3]);
    for (size_t i = (int)__argv[3]; i < sizeof(ITNN)/sizeof(*ITNN); i++)
    {
        InputFirst<<<>>>(Nptr,ITNN[i]);
        cudaDeviceSynchronize();
        for (size_t j = 1; j < Nptr->NumOfLayers; j++)
        {
            calc<<<>>>(Nptr,j);
            cudaDeviceSynchronize();
        }
        diffLast(Nptr,EFNN[i],(double)__argv[4]);
        cudaDeviceSynchronize();
        for (size_t j = Nptr->NumOfLayers; j >= 1; j--)
        {
            diffcalc<<<>>>(Nptr,j);
            cudaDeviceSynchronize();
        }
        back(Nptr);
    }
    

    cudaFree(Nptr);
    return 0;
}