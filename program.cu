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
// 6- number of parallel blocks where each one handle one of input/expected output training 
// 4- the index of last image to be trained
int main(int argc,char *argv[]){
    //int GPUId = cudaGetDevice(&GPUId);
    printf("%d is num of args\n",argc);
    if(argc != 8){
        printf("there must be 6 arguments\n");
        return -1;
    }
    int Stp = stoi(argv[4]),mlR = stod(argv[5]), pN = stoi(argv[6]), Li = stoi(argv[7]);
    // unifed (accessible from both CPU & GPU) pointer to neural network
    neuralnetwork* NNp;
    cudaError_t err = cudaMallocManaged((void**)&NNp,sizeof(neuralnetwork));
    if(err != cudaSuccess){
        printf("there is problem in allocating memory for pointer\n");
        cout << to_string(err);
        return -2;
    }
    //FromFile functoin will assign NNp with adress of neural network
    //since it's unifed memory (at least, in abstract manner), we can use GPU right away
    neuralnetwork nn = FromFile(argv[1]);
    cudaMemcpy(NNp,&nn,sizeof(nn),cudaMemcpyDefault);
    //cudaMemPrefetchAsync(NNp,sizeof(neuralnetwork),GPUId);
    
    // finding both the count of neurons of first and last layer for inputing and outputing purposes
    
    // int NumofFirst = NNp->layers[0].NumOfNu;
    // int NumofLast = NNp->layers[NNp->NumOfLayers-1].NumOfNu;

    // getting both inputs(images) & outputs(labels) for neural network
    unsigned char** ITNN;
    ITNN = InputsToNN(ITNN,argv[2],Stp);
    unsigned char** EFNN;
    string v = "";
    EFNN = ExpectedFromNN(EFNN,argv[3],Stp);
    // reapeting from starting point until reaching number of training epsidoes (which is 60,000)
    for (int i = Stp; i < Li; i+=pN)
    {
        PreBackPropagation(NNp,&ITNN[i],&EFNN[i],mlR,pN);
        cudaDeviceSynchronize();
        printf("iteration %d : error %lf\n",i,error(NNp,EFNN[i]));
        cudaDeviceSynchronize();
        back<<<284,800>>>(NNp);
        cudaDeviceSynchronize();
    }
    ToFile("version11.mn1",NNp);

    cudaFree(NNp);
    cudaFree(ITNN);
    cudaFree(EFNN);
    return 0;
}


