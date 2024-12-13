#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// #include "holders.hpp"
#include "../header files/file.hpp"
#include "../header files/mnist.hpp"
#include "../header files/calc.hu"

#define CPUbefore(layerindex){\
        for(int k=0; k<10;k++){\
            printf("value of neuron %d in layer %d in cpu before calc is %lf\n",k,layerindex,NNp->layers[layerindex].group[k].value);\
        }\
}
#define CPUafter(layerindex){\
    for(int k=0; k<10;k++){\
        printf("value of neuron %d in layer %d in cpu after calc is %lf\n",k,layerindex,NNp->layers[layerindex].group[k].value);\
    }\
}

// fun fact: argv[0] = name of the program

// arguments of main function with corresponding index
// 1- mn1 file path to read the neural network to be trained
// 2- mnist image file path
// 3- mnist label file path
// 4- the index of first image to be trained
// 5- learning rate
// 6- number of parallel blocks where each one handle one of input/expected output training 
// 7- the index of last image to be trained

__global__ void Debugkernel(neuralnetwork* nptr, double* expected){
    nptr->layers[0].group[0].value = expected[7];
}


int main(int argc,char *argv[]){
    clock_t t1,t2;
    //int GPUId = cudaGetDevice(&GPUId);
    printf("%d is num of args\n",argc);
    if(argc != 8){
        printf("there must be 7 arguments. thier number is %d\n",argc-1);
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
    
    
    // cudaMemcpy(NNp,&nn,sizeof(nn),cudaMemcpyDefault);
    *NNp = nn;
    
    //cudaMemPrefetchAsync(NNp,sizeof(neuralnetwork),GPUId);
    
    // finding both the count of neurons of first and last layer for inputing and outputing purposes
    
    // int NumofFirst = NNp->layers[0].NumOfNu;
    // int NumofLast = NNp->layers[NNp->NumOfLayers-1].NumOfNu;

    // getting both inputs(images) & outputs(labels) for neural network
    double** ITNN;
    InputsToNN(&ITNN,argv[2],Stp,Li);
    double** EFNN;
    ExpectedFromNN(&EFNN,argv[3],Stp,Li);
    cudaDeviceSynchronize();
    // reapeting from starting point until reaching number of training epsidoes (which is 60,000)
    for (int i = Stp; i <= Li; i+=pN)
    {
        t1 = clock();
        InputFirst<<<25,32>>>(NNp,ITNN[i]);
        cudaDeviceSynchronize();
        for(int  j=1; j< NNp->NumOfLayers;j++){
            calc<<<(int)pow(3,2-j),32>>>(NNp,j);
            cudaDeviceSynchronize();
        }
        diffLast<<<1,32>>>(NNp,EFNN[i],mlR);
        cudaDeviceSynchronize();
        for(int  j=NNp->NumOfLayers-2; j> 0;j--){
            diffcalc<<<(-22*j+25),32>>>(NNp,j);
            cudaDeviceSynchronize();
        }
        t2 = clock();
        printf("iteration %d : error %lf which needed %lf seconds\n",i,error(NNp,EFNN[i]),((double)t2 - (double)t1)/CLOCKS_PER_SEC);
        cudaDeviceSynchronize();
        back<<<397,160>>>(NNp);
        cudaDeviceSynchronize();
    }
    cudaFree(ITNN);
    cudaFree(EFNN);
    ToFile("version11.mn1",NNp);

    cudaFree(NNp);
    return 0;
}


