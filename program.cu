#include <stdio.h>
#include <stdlib.h>
// #include "holders.hpp"
#include "file.hpp"
#include "mnist.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include "calc.hu"
// fun fact: argv[0] = name of the program

// arguments of main function with corresponding index
// 1- mn1 file path to read the neural network to be trained
// 2- mnist image file path
// 3- mnist label file path
// 4- the index of first image to be trained
// 5- learning rate


int main(int argc,char *argv[]){
    int GPUId = cudaGetDevice(&GPUId);
    printf("%d is num of args and. also %d is id of gpu\n",argc,GPUId);
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
    neuralnetwork nn = FromFile(argv[1]);
    //NNp = &nn;
    cudaMemcpy(NNp,&nn,sizeof(nn),cudaMemcpyDefault);
    cudaMemPrefetchAsync(NNp,sizeof(neuralnetwork),GPUId);
    
    // finding both the count of neurons of first and last layer for inputing and outputing purposes
    
    // int NumofFirst = NNp->layers[0].NumOfNu;
    // int NumofLast = NNp->layers[NNp->NumOfLayers-1].NumOfNu;

    // getting both inputs(images) & outputs(labels) for neural network
    unsigned char** ITNN;
    ITNN = InputsToNN(ITNN,argv[2],(int)stoi(argv[4]));
    unsigned char** EFNN;
    string v = "";
    EFNN = ExpectedFromNN(EFNN,argv[3],(int)stoi(argv[4]));
    // reapeting from starting point until reaching number of training epsidoes (which is 60,000)
    for (int i = (int)stoi(argv[4]); i < 10000; i++)
    {
        // InputFirst<<<>>>(NNp, ITNN[i]);
        // cudaDeviceSynchronize();
        // for (size_t j = 1; j < NNp->NumOfLayers; j++)
        // {
        //     calc<<<>>>(NNp, j);
        //     cudaDeviceSynchronize();
        // }
        // diffLast(NNp, (unsigned char *)EFNN[i], stod(argv[5]));
        // cudaDeviceSynchronize();
        // for (size_t j = NNp->NumOfLayers; j >= 1; j--)
        // {
        //     diffcalc<<<>>>(NNp, j);
        //     cudaDeviceSynchronize();
        // }
        preback<<<1,784>>>(NNp,ITNN[i],EFNN[i],stod(argv[5]));
        cudaDeviceSynchronize();
        printf("iteration %d : error %f\n",i,error(NNp,EFNN[i]));
        cin >> v;
        back<<<289,784>>>(NNp);
        cudaDeviceSynchronize();
    }
    

    cudaFree(NNp);
    cudaFree(ITNN);
    cudaFree(EFNN);
    return 0;
}


