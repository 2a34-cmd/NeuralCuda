#include "file.cpp"
#include <math.h>

__global__ void calc(struct neuralnetwork* neuralnetptr,int layerIndex,double* inputs,double* outputs){
    int j = blockDim.x * blockIdx.x + threadIdx.x;//index of neuron

    neuralnetwork NN = *neuralnetptr;
    connection* conptr;
    NuCon Froms = NN.layers[layerIndex].group[j].froms;
    cudaMalloc((void**)&conptr,Froms.NumOfCon*sizeof(connection));

    double weitghedSum = 0;
    for (int k = 0; k < Froms.NumOfCon; k++)
    {

        weitghedSum+=  *  inputs[k];
    }
    weitghedSum += NN.layers[layerIndex].group[j].bias;
    Activation(&weitghedSum,NN.ActivFunc,&(outputs[j]));
    NN.layers[layerIndex].group[j].value = outputs[j];
    cudaFree(conptr);
}
__global__ void diffcalc(neuralnetwork* neuralnetptr,int layerIndex,double* Expected,double* outputs){
    int i = blockDim.x * blockIdx.x + threadIdx.x;//index of neuron


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
__device__ void DActivation(double* input,ActivationFunc AF,double* output){
    switch(AF){
    case 1:
        double out = pow(cosh(*input),-2);
        output = &out;
        break;
    case 2:
        double out = exp(*input)*pow(1+exp(*input),-2);
        output = &out;
        break;
    case 3:
        double out=0;
        if(*input > 0) out = 1;
        output = &out;
        break;
    case 4:
        double out = 1;
        output = &out;
        break;
    default:
        break;
    }
}