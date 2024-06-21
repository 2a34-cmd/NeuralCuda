#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "holders.hpp"
#include "mnist.hpp"
#include "file.hpp"
#include "calc.cu"
// fun fact: __argv[0] = name of the program

// arguments of main function with corresponding index
// 1- mn1 file path to read the neural network to be trained
// 2- mnist image file path
// 3- mnist label file path
// 4- the index of first image to be trained
// 5- learning rate
int main(int __argc,char *__argv[]){
    if(__argc != 6){
        printf("there must be 5 arguments");
        return -1;
    }

    // unifed (accessible from both CPU & GPU) pointer to neural network 
    neuralnetwork* NNp;
    cudaError_t err = cudaMallocManaged((void**)&NNp,sizeof(neuralnetwork));
    if(err != cudaSuccess){
        cout << "there is problem in allocating memory for pointer\n";
        cout << to_string(err);
        return -2;
    }
    //FromFile functoin will assign NNp with adress of neural network
    //since it's unifed memory (at least, in abstract manner), we can use GPU right away
    FromFile(__argv[1],NNp);
    
    // finding both the count of neurons of first and last layer for inputing and outputing purposes
    int NumofFirst = NNp->layers[0].NumOfNu;
    int NumofLast = NNp->layers[NNp->NumOfLayers-1].NumOfNu;

    // getting both inputs(images) & outputs(labels) for neural network
    byte** ITNN = InputsToNN(__argv[2],(int)__argv[4]);
    byte** EFNN = ExpectedFromNN(__argv[3],(int)__argv[4]);

    // reapeting from starting point until reaching number of training epsidoes (which is 60,000)
    for (size_t i = (int)__argv[4]; i < 60000; i++)
    {
        // InputFirst<<<>>>(NNp, ITNN[i]);
        // cudaDeviceSynchronize();
        // for (size_t j = 1; j < NNp->NumOfLayers; j++)
        // {
        //     calc<<<>>>(NNp, j);
        //     cudaDeviceSynchronize();
        // }
        // diffLast(NNp, (byte *)EFNN[i], stod(__argv[5]));
        // cudaDeviceSynchronize();
        // for (size_t j = NNp->NumOfLayers; j >= 1; j--)
        // {
        //     diffcalc<<<>>>(NNp, j);
        //     cudaDeviceSynchronize();
        // }


        
        preback<<<,1>>>(NNp,ITNN[i],EFNN[i],stod(__argv[5]));
        back<<<,>>>(NNp);
    }
    

    cudaFree(NNp);
    return 0;
}