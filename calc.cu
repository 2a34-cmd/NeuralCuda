#include <math.h>
#include "holders.hpp"

/// The functions below don't use the idea of child threads, which may be implemented later

// this function needs <<<N,M>>> where N*M equals number of neuronals in the layer with layerIndex  
// can't use with layerIndex == 0
/// to call it, you must call InputFirst
__global__ void calc(neuralnetwork* neuralnetptr,int layerIndex){
    int j = blockDim.x * blockIdx.x + threadIdx.x;//index of neuron
    neuralnetwork NN = *neuralnetptr;
    NuCon Froms = NN.layers[layerIndex].group[j].froms;
    double weightedSum = NN.layers[layerIndex].group[j].bias;
    for (size_t i = 0; i < Froms.NumOfCon; i++)
    {
        weightedSum += Froms.ConPtr[i]->weight * NN.layers[Froms.ConPtr[i]->LF].group[Froms.ConPtr[i]->FromId].value;
    }
    Activation(weightedSum,NN.ActivFunc,&NN.layers[layerIndex].group[j].value);
    
}
// this function needs <<<N,M>>> where N*M equals number of neuronals in the layer with layerIndex
// can't use the function with layerIndex +1 == neuralnetptr->numOfLayers
/// to call it, you must call difflast and calc
__global__ void diffcalc(neuralnetwork* neuralnetptr,int layerIndex){
    int j = blockDim.x * blockIdx.x + threadIdx.x;//index of neuron

    neuralnetwork NN = *neuralnetptr;
    connection* conptr;
    NuCon Toes = NN.layers[layerIndex].group[j].toes;
    double value = NN.layers[layerIndex].group[j].value;
    double weightedSum = 0;
    double Term = 1;
    for (size_t i = 0; i < Toes.NumOfCon; i++)
    {
        Term *= DActivation(Toes.ConPtr[i]->weight * value + NN.layers[Toes.ConPtr[i]->LT].group[Toes.ConPtr[i]->ToId].bias,
                            NN.ActivFunc);
        Term *= NN.layers[Toes.ConPtr[i]->LT].group[Toes.ConPtr[i]->ToId].difference;
        Term *= Toes.ConPtr[i]->weight;
        weightedSum += Term;
        Term = 1;
    }
    NN.layers[layerIndex].group[j].difference = weightedSum;
}


__global__ void InputFirst(neuralnetwork* neuralnetptr, double* Inputs){
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    neuralnetptr->layers[0].group[j].value = Inputs[j];
}
__global__ void diffLast(neuralnetwork* neuralnetptr,double* Expected, double MLRate){
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    neuron n = neuralnetptr->layers[neuralnetptr->NumOfLayers-1].group[j];
    n.difference = MLRate * abs(n.value - Expected[j]);
}

//<<<N,M>>> where N*M == number of neurons and connections
/// to call it, you must call cycle and its formers
__global__ void back(neuralnetwork* neuralnetptr){
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    neuralnetwork NN = *neuralnetptr;
    if(j<NN.NumOfConnenction){
        neuron n = NN.layers[NN.connections->LT].group[NN.connections->ToId];
        double diff = n.difference,val = n.value;
        NN.connections[j].weight -= diff*val;
    }
    else{   
        j -= NN.NumOfConnenction;
        int k =j;
        int x = j,y=0;
        for (size_t i = 0; i < NN.NumOfLayers; i++)
        {
            if(NN.layers[i].NumOfNu < k){
                k-= NN.layers[i].NumOfNu;
                x=k;
                y++;
            }else{
                x--;
                break;
            }
        }
        NN.layers[y].group[x].bias -= NN.layers[y].group[x].difference;
    }
}


//the activation function and its derviative
__device__ void Activation(double input,ActivationFunc af,double* output){
    switch (af)
    {
    case 1:
        double out = tanh(input);
        output = &out;
        break;
    case 2:
        double out = exp(input)/(1+exp(input));
        output = &out;
        break;
    case 3:
        double out=0;
        if(input > 0) out = input;
        output = &out;
        break;
    case 4:
        output = &input;
        break;
    default:
        break;
    }
}
__device__ double DActivation(double input,ActivationFunc AF){
    switch(AF){
    case 1:
        return pow(cosh(input),-2);
    case 2:
        return exp(input)*pow(1+exp(input),-2);
    case 3:
        if(input > 0) return 1;
        else return 0;
    case 4:
        return 1;
    default:
        return NAN;
    }
}