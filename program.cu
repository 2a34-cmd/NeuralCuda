#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
// #include "holders.hpp"
#include "file.hpp"
#include "mnist.hpp"
#include "calc.hu"
// fun fact: argv[0] = name of the program

// arguments of main function with corresponding index
// 1- mn1 file path to read the neural network to be trained
// 2- mnist image file path
// 3- mnist label file path
// 4- the index of first image to be trained
// 5- learning rate


int main(int argc,char *argv[]){
    //int GPUId = cudaGetDevice(&GPUId);
    printf("%d is num of args\n",argc);
    // if(argc != 6){
    //     printf("there must be 5 arguments\n");
    //     return 5;
    // }

    // unifed (accessible from both CPU & GPU) pointer to neural network
    neuralnetwork* NNp;
    cudaError_t err = cudaMallocManaged((void**)&NNp,sizeof(neuralnetwork));
    if(err != cudaSuccess){
        cout << "there is problem in allocating memory for pointer\n";
        cout << to_string(err);
        return 3;
    }
    //FromFile functoin will assign NNp with adress of neural network
    //since it's unifed memory (at least, in abstract manner), we can use GPU right away
    neuralnetwork nn = FromFile("version10.mn1");
    //NNp = &nn;
    cudaMemcpy(NNp,&nn,sizeof(nn),cudaMemcpyDefault);
    //cudaMemPrefetchAsync(NNp,sizeof(neuralnetwork),GPUId);
    
    // finding both the count of neurons of first and last layer for inputing and outputing purposes
    
    // int NumofFirst = NNp->layers[0].NumOfNu;
    // int NumofLast = NNp->layers[NNp->NumOfLayers-1].NumOfNu;

    // getting both inputs(images) & outputs(labels) for neural network
    unsigned char** ITNN;
    ITNN = InputsToNN(ITNN,"img.idx3",0);
    unsigned char** EFNN;
    string v = "";
    EFNN = ExpectedFromNN(EFNN,"lbl.idx1",0);
    // reapeting from starting point until reaching number of training epsidoes (which is 60,000)
    for (int i = 0; i < 10000; i++)
    {
        InputFirst<<<28,28>>>(NNp, ITNN[i]);
        cudaDeviceSynchronize();
        for (size_t j = 1; j < NNp->NumOfLayers; j++)
        {
            calc<<<8,32>>>(NNp, j);
            cudaDeviceSynchronize();
        }
        printf("iteration %d : error %f\n",i,error(NNp,EFNN[i]));
        cudaDeviceSynchronize();
        diffLast<<<1,10>>>(NNp, (unsigned char *)EFNN[i], 2.0);
        cudaDeviceSynchronize();
        for (size_t j = NNp->NumOfLayers -2; j >= 1; j--)
        {
            diffcalc<<<25,32>>>(NNp, j);
            cudaDeviceSynchronize();
        }
        // preback<<<1,784>>>(NNp,ITNN[i],EFNN[i],0.5);
        // cin >> v;
        back<<<284,800>>>(NNp);
        cudaDeviceSynchronize();
    }
    

    cudaFree(NNp);
    cudaFree(ITNN);
    cudaFree(EFNN);
    return 0;
}


