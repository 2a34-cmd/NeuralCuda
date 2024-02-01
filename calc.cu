#include "file.cpp"
#include <math.h>

__global__ void calc(struct neuralnetwork* neuralnetptr,int layerIndex,double* inputs,double* outputs){
    int i = threadIdx.x;//index of connection
    int j = blockIdx.x;//index of nueron

    neuralnetwork NN = *neuralnetptr;
    connection* conptr;
    int* Froms;
    FromsOfNeuron<<<2,112901>>>(conptr,layerIndex,j,Froms);
    cudaMalloc((void**)&conptr,(*Froms)*sizeof(connection));

    double weitghedSum = 0;
    for (int k = 0; k < *Froms; k++)
    {
        weitghedSum+= conptr[i].weight * inputs[i];
    }
    double* param = outputs + (sizeof(double) * j);
    Activation(&weitghedSum,NN.ActivFunc,param);
    NN.layers[layerIndex].group[j].value = *param;
    cudaFree(Froms);
}
__global__ void train(neuralnetwork* neuralnetptr,double* inputs,double* outputs){
    int i = threadIdx.x;
    int j = blockIdx.x;
    int globalindex = blockDim.x * j + i;


}

__device__ void FromsOfNeuron(connection* conptr,unsigned int LId,unsigned int NId,int* output){
    int i = threadIdx.x;
    int j = blockIdx.x;
    int globalindex = blockDim.x * j + i;
    if(conptr[globalindex].LT == LId && conptr[globalindex].ToId == NId){
        (*output)++;
    }
}
__device__ void ToesOfNeuron(connection* conptr,unsigned int LId,unsigned int NId,int* output){
    int i = threadIdx.x;
    int j = blockIdx.x;
    int globalindex = blockDim.x * j + i;
    if(conptr[globalindex].LF == LId && conptr[globalindex].FromId == NId){
        (*output)++;
    }
}
__device__ void Activation(double* input,ActivationFunc af,double* output){
    switch (af)
    {
    case 1:
        double out = tanh(*input);
        output = &out;
        break;
    case 2:
        double out = exp(*input)/(1+exp(*input));
        output = &out;
        break;
    case 3:
        double out=0;
        if(*input > 0) out = *input;
        output = &out;
        break;
    case 4:
        output = input;
        break;
    default:
        break;
    }
}