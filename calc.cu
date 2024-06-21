#include <math.h>
#include "holders.hpp"
using namespace std;
/// The functions below don't use the idea of child threads, which may be implemented later

// this function needs <<<N,M>>> where N*M equals number of neuronals in the layer with layerIndex  
// can't use with layerIndex == 0
/// to call it, you must call InputFirst
__global__ void calc(neuralnetwork* neuralnetptr,int layerIndex){
    int j = blockDim.x * blockIdx.x + threadIdx.x;//index of neuron
    neuralnetwork NN = *neuralnetptr;
    NuCon Froms = NN.layers[layerIndex].group[j].froms;
    double weightedSum = NN.layers[layerIndex].group[j].bias; //initilized with neuron bias
    for (size_t i = 0; i < Froms.NumOfCon; i++)
    {
        weightedSum += (Froms.ConPtr[i]->weight) * NN.layers[Froms.ConPtr[i]->LF].group[Froms.ConPtr[i]->FromId].value;
    }
    Activation(weightedSum,NN.ActivFunc,&NN.layers[layerIndex].group[j].value);
    
}
// this function needs <<<N,M>>> where N*M equals number of neuronals in the layer with layerIndex
// can't use the function with layerIndex +1 == neuralnetptr->numOfLayers
/// to call it, you must call difflast and calc
__global__ void diffcalc(neuralnetwork* neuralnetptr,int layerIndex){
    int j = blockDim.x * blockIdx.x + threadIdx.x;//index of neuron


    //to calculate the difference, we are using chain rule with adding all the possible terms
    // so we will have term wich will be (del error/del this neuron)
    // it will be recurrence relation when using chain rule
    // by chain rule it can be (del error/del next neuron)* (del next neuron/ del this neuron)
    // the second ratio is Derviative of activationfunction(next neuron wieghted sum) * wieght of 
    //          connection from this neuron to next neuron
    // the wieghted sum is then the sum of all terms of next neurons

    // the equation is (del error/del neuron value) = SUM{0<=i<num of Toes connection}( 
    //                     (del error/del finishing neuron) * 
    //                     DActivation(inverse of activation(finishing neuron value)) * 
    //                     wieght from this neuron to finishing neuron   )
    neuralnetwork NN = *neuralnetptr;
    connection* conptr;
    NuCon Toes = NN.layers[layerIndex].group[j].toes;// the list of connections to next neurons
    double value = NN.layers[layerIndex].group[j].value;// value of this neuron
    double weightedSum = 0;
    double Term = 1;
    for (size_t i = 0; i < Toes.NumOfCon; i++)// we are summing over connections
    {
        // to calculate the wieghted sum of next neuron, it is recommended to fins its Froms connections
        NuCon froms = NN.layers[Toes.ConPtr[i]->LT].group[Toes.ConPtr[i]->ToId].froms;

        double LinearExp = NN.layers[Toes.ConPtr[i]->LT].group[Toes.ConPtr[i]->ToId].bias;
        for (size_t j = 0; j < froms.NumOfCon; j++)
        {
            LinearExp += (froms.ConPtr[j]->weight) * NN.layers[froms.ConPtr[j]->LF].group[froms.ConPtr[j]->FromId].value;
        }
        // linear expersion should be the input of Daf
        Term *= DActivation(LinearExp,NN.ActivFunc);
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
__global__ void InputFirst(neuralnetwork* neuralnetptr, byte* Inputs){
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    neuralnetptr->layers[0].group[j].value = (double)Inputs[j];
}


__global__ void diffLast(neuralnetwork* neuralnetptr,double* Expected, double MLRate){
    int j = blockDim.x * blockIdx.x + threadIdx.x;// indexing neurons
    neuron n = neuralnetptr->layers[neuralnetptr->NumOfLayers-1].group[j];
    n.difference = MLRate * (n.value - Expected[j]);
}
__global__ void diffLast(neuralnetwork* neuralnetptr,byte* Expected, double MLRate){
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    neuron n = neuralnetptr->layers[neuralnetptr->NumOfLayers-1].group[j];
    n.difference = MLRate * (n.value - (double)Expected[j]);
}


// we need to call InputFirst, (serial) calc, diffLast before calling back
// this method does that without caring much about parallelizing in best shape
//      since the methods are fast calculations
//<<<N,M>>> where N*M = max number of neurons in a layer
//  since we need to sync the threads (b/c sometimes group[j] is null due to design decission)
//      M needs to be 1 
__global__ void preback(neuralnetwork* neuralnetptr, double* Inputs,double* Expected, double MLRate){
    int j = threadIdx.x;// indexing neurons
    try{
    neuralnetptr->layers[0].group[j].value = Inputs[j];// finished InputFirst
    }catch(...){}
    
    neuralnetwork NN = *neuralnetptr;
    for(size_t i = 1 ;i<NN.NumOfLayers;i++){
        __syncthreads();
        try{
            NuCon Froms = NN.layers[i].group[j].froms;
            double weightedSum = NN.layers[i].group[j].bias; //initilized with neuron bias
            for (size_t i = 0; i < Froms.NumOfCon; i++)
            {
                weightedSum += (Froms.ConPtr[i]->weight) * NN.layers[Froms.ConPtr[i]->LF].group[Froms.ConPtr[i]->FromId].value;
            }
            Activation(weightedSum,NN.ActivFunc,&NN.layers[i].group[j].value);
        }
        catch(...){}
    }//finished calc funcitons
    __syncthreads();
    try{
        neuron n = neuralnetptr->layers[neuralnetptr->NumOfLayers-1].group[j];
        n.difference = MLRate * (n.value - Expected[j]);
    }
    catch(...){}//finished DiffLast

    __syncthreads();// it might be not needed, but I'm not sure
}
__global__ void preback(neuralnetwork* neuralnetptr, byte* Inputs,byte* Expected, double MLRate){
    int j = threadIdx.x;// indexing neurons
    try{
    neuralnetptr->layers[0].group[j].value = (double)Inputs[j];// finished InputFirst
    }catch(...){}
    
    neuralnetwork NN = *neuralnetptr;
    for(size_t i = 1 ;i<NN.NumOfLayers;i++){
        __syncthreads();
        try{
            NuCon Froms = NN.layers[i].group[j].froms;
            double weightedSum = NN.layers[i].group[j].bias; //initilized with neuron bias
            for (size_t i = 0; i < Froms.NumOfCon; i++)
            {
                weightedSum += (Froms.ConPtr[i]->weight) * NN.layers[Froms.ConPtr[i]->LF].group[Froms.ConPtr[i]->FromId].value;
            }
            Activation(weightedSum,NN.ActivFunc,&NN.layers[i].group[j].value);
        }
        catch(...){}
    }//finished calc funcitons
    __syncthreads();
    try{
        neuron n = neuralnetptr->layers[neuralnetptr->NumOfLayers-1].group[j];
        n.difference = MLRate * (n.value - (double)Expected[j]);
    }
    catch(...){}//finished DiffLast
    
    __syncthreads();// it might be not needed, but I'm not sure
}

//<<<N,M>>> where N*M == number of connections
/// to call it, you must call cycle and its formers
__global__ void back(neuralnetwork* neuralnetptr){
    int j = blockDim.x * blockIdx.x + threadIdx.x;// indexing wieghts and biases
    neuralnetwork NN = *neuralnetptr;
    if(j<NN.NumOfConnenction){ // checking if j in in range of wieghts indeices
        neuron n = NN.layers[NN.connections->LT].group[NN.connections->ToId];
        double diff = n.difference;
        double val = NN.layers[NN.connections->LF].group[NN.connections->FromId].value;

        double LinearExp = n.bias;
        for (size_t j = 0; j < n.froms.NumOfCon; j++)
        {
            LinearExp += (n.froms.ConPtr[j]->weight) * NN.layers[n.froms.ConPtr[j]->LF].group[n.froms.ConPtr[j]->FromId].value;
        }
        // (del error/del wieght) = (del error/ del finishing neuron)(del finishing neuron /del wiegth)
        //                        = finishing neuron difference * starting neuron value *   
        //                          Daf(linear exp of finishing nuron) 
        NN.connections[j].weight -= diff*val* DActivation(LinearExp,NN.ActivFunc);
        // (del error/del bias) = (del error/del neuron) * (del neuron/del bias)
        //                      = neuron difference      * Daf(linear exp)
        // however, this command below will be repeated in number of connections that have n 
        //                                                               as finishing neuron
        // so to normalize it, it comes the idea to multiply the experssion below by
        //                     1/(num of froms of n)
        n.bias -= diff *DActivation(LinearExp,NN.ActivFunc) /(n.froms.NumOfCon);
    }
}


//the activation function and its derviative
__device__ void Activation(double input,ActivationFunc af,double* output){
    switch (af)
    {
    case 1: //hyperbolic tangent
        double out = tanh(input);
        output = &out;
        break;
    case 2: //sigmoid (the same function of  FD distribution) 1/(e^(-x) +1)
        double out = exp(input)/(1+exp(input));
        output = &out;
        break;
    case 3: // rectified linear unit : if positive, like linear fuction. otherwise, it's 0
        double out=0;
        if(input > 0) out = input;
        output = &out;
        break;
    case 4:// identity function. it's nonlinear, but it's here for CNN and other types
        output = &input;
        break;
    default:
        break;
    }
}
// this method is just returning the value of derviative of activation function, given its code AF and an input
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
