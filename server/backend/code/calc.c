#include "../header/file.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
double Activation(double input, ActivationFunc af)
{
    double out;
    switch (af)
    {
    case 1: // hyperbolic tangent
        out = tanh(input);
        return out;
    case 2: // sigmoid (the same function of  FD distribution) 1/(e^(-x) +1)
        out = exp(input) / (1 + exp(input));
        return out;
    case 3: // rectified linear unit : if positive, like linear fuction. otherwise, it's 0
        out = 0;
        if (input > 0)
            out = input;
        return out;
    case 4: // identity function. it's nonlinear, but it's here for CNN and other types
        return input;
    default:
        return 0;
    }
}

void calc( int layerIndex)
{
    for (int j=0; j < NN.layers[layerIndex].NumOfNu; j++)
    {
        NuCon Toes = NN.layers[layerIndex].group[j].toes;
        double weightedSum = NN.layers[layerIndex].group[j].bias; // initilized with neuron bias
        for (int i = 0; i < Toes.NumOfCon; i++)
        {
            weightedSum += ((*Toes.ConPtr[i]).weight) * NN.layers[(*Toes.ConPtr[i]).LF].group[(*Toes.ConPtr[i]).FromId].value;
        }
        NN.layers[layerIndex].group[j].value = Activation(weightedSum, NN.ActivFunc);
    }
}

//no side effect on neural network
void calcNoSE( int layerIndex,double* Inputs,double* Outputs)
{
    for (int j=0; j < NN.layers[layerIndex].NumOfNu; j++)
    {
        NuCon Toes = NN.layers[layerIndex].group[j].toes;
        double weightedSum = NN.layers[layerIndex].group[j].bias; // initilized with neuron bias
        for (int i = 0; i < Toes.NumOfCon; i++)
        {
            connection conn = *Toes.ConPtr[i];
            weightedSum += (conn.weight) * Inputs[conn.FromId];
        }
        Outputs[j] = Activation(weightedSum, NN.ActivFunc);
    }
}


void InputFirst( unsigned char *Inputs)
{
    for(int j =0; j<NN.layers[0].NumOfNu; j++){
    NN.layers[0].group[j].value = (double)(Inputs[j]) / 128 - 1;
    }
}

//no side effect on neuralnetwork
void InputFirstNoSE( unsigned char *Inputs,double* Outputs)
{
    for(int j =0; j<NN.layers[0].NumOfNu; j++){
        Outputs[j] = (double)(Inputs[j]) / 128 - 1;
    }
}

void CalcNeuralNetwork(unsigned char *Inputs,double* Outputs){
    InputFirst(Inputs);
    double sum = 0;
    for(int i=0; i <NN.NumOfLayers;i++){
        calc(i);
    }
    for(int i=0; i < NN.layers[NN.NumOfLayers -1].NumOfNu; i++){
        sum += Outputs[i];
    }
    for(int i=0; i < NN.layers[NN.NumOfLayers -1].NumOfNu; i++){
        Outputs[i] = (Outputs[i]/sum) * 100;
    }
    return;
}

//no side effect on neural network
double* CalcNeuralNetworkNoSE(unsigned char *Inputs){
    double* Alternative = malloc((NN.layers[0].NumOfNu)*sizeof(double));
    double* Alternative2 = NULL;
    InputFirstNoSE(Inputs,Alternative);
    uint8_t WhichIsOutput = 1;
    double sum = 0;
    for(int i=1; i <NN.NumOfLayers;i+=2){
        Alternative2 = realloc(Alternative2,(NN.layers[i].NumOfNu)*sizeof(double));
        calcNoSE(i,Alternative,Alternative2);
        WhichIsOutput = 2;
        free(Alternative);
        if(i+1 < NN.NumOfLayers){
            Alternative = realloc(Alternative,(NN.layers[i+1].NumOfNu)*sizeof(double));
            calcNoSE(i+1,Alternative2,Alternative);
            WhichIsOutput = 1;
            free(Alternative2);
        }
    }
    if(WhichIsOutput == 2){
        Alternative = realloc(Alternative,(NN.layers[NN.NumOfLayers -1].NumOfNu)*sizeof(double));
        memcpy(Alternative,Alternative2,(NN.layers[NN.NumOfLayers -1].NumOfNu)*sizeof(double));
        free(Alternative2);
    }
    for(int i=0; i < NN.layers[NN.NumOfLayers -1].NumOfNu; i++){
        sum += Alternative[i];
    }
    for(int i=0; i < NN.layers[NN.NumOfLayers -1].NumOfNu; i++){
        Alternative[i] /= sum;
        Alternative[i] *= 100;
    }
    return Alternative;
}