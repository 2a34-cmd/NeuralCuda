#include <math.h>
#include "holders.hpp"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
using namespace std;

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        const char* s = cudaGetErrorString(code);

        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// the activation function and its derviative
__device__ double Activation(double input, ActivationFunc af)
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
        return NAN;
    }
}
// this method is just returning the value of derviative of activation function, given its code AF and an input
__device__ double DActivation(double input, ActivationFunc AF)
{
    switch (AF)
    {
    case 1:
        return pow(cosh(input), -2);
    case 2:
        return exp(input) * pow(1 + exp(input), -2);
    case 3:
        if (input > 0)
            return 1;
        else
            return 0;
    case 4:
        return 1;
    default:
        return NAN;
    }
}

/// The functions below don't use the idea of child threads, which may be implemented later

// this function needs <<<N,M>>> where N*M equals number of neuronals in the layer with layerIndex
// can't use with layerIndex == 0
/// to call it, you must call InputFirst
__global__ void calc(neuralnetwork *neuralnetptr, int layerIndex)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x; // index of neuron
    neuralnetwork NN = *neuralnetptr;
    if (j < NN.layers[layerIndex].NumOfNu)
    {
        NuCon Toes = NN.layers[layerIndex].group[j].toes;
        double weightedSum = NN.layers[layerIndex].group[j].bias; // initilized with neuron bias
        for (int i = 0; i < Toes.NumOfCon; i++)
        {
            weightedSum += (Toes.ConPtr[i]->weight) * NN.layers[Toes.ConPtr[i]->LF].group[Toes.ConPtr[i]->FromId].value;
        }
        NN.layers[layerIndex].group[j].value = Activation(weightedSum, NN.ActivFunc);
    }
    // neuralnetptr = &NN;
}
// this function needs <<<N,M>>> where N*M equals number of neuronals in the layer with layerIndex
// can't use the function with layerIndex == neuralnetptr->numOfLayers -1
// b/c it is the last layer and has special function diffLast
/// to call it, you must call difflast and calc
__global__ void diffcalc(neuralnetwork *neuralnetptr, int layerIndex)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x; // index of neuron

    // to calculate the difference, we are using chain rule with adding all the possible terms
    //  so we will have term wich will be (del error/del this neuron)
    //  it will be recurrence relation when using chain rule
    //  by chain rule it can be (del error/del next neuron)* (del next neuron/ del this neuron)
    //  the second ratio is Derviative of activationfunction(next neuron wieghted sum) * wieght of
    //           connection from this neuron to next neuron
    //  the wieghted sum is then the sum of all terms of next neurons

    // the equation is (del error/del neuron value) = SUM{0<=i<num of Toes connection}(
    //                     (del error/del finishing neuron) *
    //                     DActivation(inverse of activation(finishing neuron value)) *
    //                     wieght from this neuron to finishing neuron   )
    if (j < neuralnetptr->layers[layerIndex].NumOfNu)
    {
        neuralnetwork NN = *neuralnetptr;
        // connection* conptr;
        NuCon Froms = NN.layers[layerIndex].group[j].froms; // the list of connections to next neurons
        // double value = NN.layers[layerIndex].group[j].value;// value of this neuron
        double weightedSum = 0;
        double Term = 1;
        for (int i = 0; i < Froms.NumOfCon; i++) // we are summing over connections
        {
            // to calculate the wieghted sum of next neuron, it is recommended to fins its Froms connections
            NuCon toes = NN.layers[Froms.ConPtr[i]->LT].group[Froms.ConPtr[i]->ToId].toes;

            double LinearExp = NN.layers[Froms.ConPtr[i]->LT].group[Froms.ConPtr[i]->ToId].bias;
            for (int k = 0; k < toes.NumOfCon; k++)
            {
                LinearExp += (toes.ConPtr[k]->weight) * NN.layers[toes.ConPtr[k]->LF].group[toes.ConPtr[k]->FromId].value;
            }
            // linear expersion should be the input of Daf
            Term *= DActivation(LinearExp, NN.ActivFunc);
            Term *= NN.layers[Froms.ConPtr[i]->LT].group[Froms.ConPtr[i]->ToId].difference;
            Term *= Froms.ConPtr[i]->weight;
            weightedSum += Term;
            Term = 1;
        }
        NN.layers[layerIndex].group[j].difference = weightedSum;
    }
}

__global__ void InputFirst(neuralnetwork *neuralnetptr, double *Inputs)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    neuralnetptr->layers[0].group[j].value = Inputs[j];
}
__global__ void InputFirst(neuralnetwork *neuralnetptr, unsigned char *Inputs)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    neuralnetptr->layers[0].group[j].value = (double)(Inputs[j]) / 128 - 1;
}

__global__ void diffLast(neuralnetwork *neuralnetptr, double *Expected, double MLRate)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x; // indexing neurons
    neuralnetptr->layers[neuralnetptr->NumOfLayers - 1].group[j].difference =
        MLRate * (neuralnetptr->layers[neuralnetptr->NumOfLayers - 1].group[j].value - Expected[j]);
}
__global__ void diffLast(neuralnetwork *neuralnetptr, unsigned char *Expected, double MLRate)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    neuralnetptr->layers[neuralnetptr->NumOfLayers - 1].group[j].difference =
        MLRate * (neuralnetptr->layers[neuralnetptr->NumOfLayers - 1].group[j].value - (double)Expected[j]);
}

// we need to call InputFirst, (serial) calc, diffLast before calling back
//<<<N,M,L>>> where N = number of parallelization, or size of inputs
// while M = max(number of neurons in layer)
//  lastly, L = sizeof(double)*2*number of neurons in *nueralnetptr
__global__ void preback(neuralnetwork *neuralnetptr, double **Inputs, double **Expected, double MLRate, double *globaldiff, double *globalval)
{
    extern __shared__ char s[]; // using char b/c sizeof(char) = 1 byte
    int *count = (int *)s;      // the same meaning as &s[0]
    int j = threadIdx.x;        // indexing neurons
    int l = blockIdx.x;         // indexing inputs and expected
    // values is 1D array where element values[(number of neurons in previous layers to layer i)
    //       + j] is the value
    //   of neuron j in layer i using inputs[l]
    // differences is sorted in the same manner for difference values of neurons
    // count will have the size of number of layers of NN and
    //   each element = offset in values and differences for layers.
    // count job is for caching layer offset as it will be used a lot.
    // globalval and globaldiff are arrays in global memory that hold the values and differences
    //  of each block shared memory in ordered shape.
    neuralnetwork NN = *neuralnetptr;
    if (j <= NN.NumOfLayers)
    {
        int v = 0;
        for (int i = 0; i < j; i++)
        {
            v += NN.layers[i].NumOfNu;
        }
        count[j] = v;
    } // finished caching
    __syncthreads();
    double *values = (double *)(sizeof(int) * (NN.NumOfLayers + 1) + s); //
    double *differences = (double *)(sizeof(double) * count[NN.NumOfLayers] + sizeof(int) * (NN.NumOfLayers + 1) + s);
    if (j < NN.layers[0].NumOfNu)
    {
        values[j] = Inputs[l][j];
        globalval[l * count[NN.NumOfLayers] + j] = values[j];
    } // finished InputFirst

    for (int i = 1; i < NN.NumOfLayers; i++)
    {
        __syncthreads();
        if (j < NN.layers[i].NumOfNu)
        {
            NuCon Toes = NN.layers[i].group[j].toes;
            double weightedSum = NN.layers[i].group[j].bias; // initilized with neuron bias
            for (int k = 0; k < Toes.NumOfCon; k++)
            {
                weightedSum += (Toes.ConPtr[k]->weight) * values[count[Toes.ConPtr[k]->LF] + Toes.ConPtr[k]->FromId];
            }
            values[count[i] + j] = Activation(weightedSum, NN.ActivFunc);
            globalval[l * count[NN.NumOfLayers] + count[i] + j] = values[count[i] + j];
        }
    } // finished calc funcitons
    __syncthreads();
    if (j < NN.layers[NN.NumOfLayers - 1].NumOfNu)
    {
        differences[count[NN.NumOfLayers - 1] + j] = MLRate *
                                                     (values[count[NN.NumOfLayers - 1] + j] - Expected[l][j]);
        globaldiff[l * count[NN.NumOfLayers] + count[NN.NumOfLayers] + j] = differences[count[NN.NumOfLayers] + j];
    } // finished difflast
    for (int i = NN.NumOfLayers - 2; i > 0; i--)
    {
        __syncthreads();
        if (j < NN.layers[i].NumOfNu)
        {
            double weightedSum = 0;
            NuCon Froms = NN.layers[i].group[j].froms; // the one used for back probagation
            for (int n = 0; n < Froms.NumOfCon; n++)
            {
                double Term = 1;
                double LinearExp = NN.layers[i].group[j].bias;
                NuCon Toes = NN.layers[i].group[j].toes; // the one used for calc
                for (int k = 0; k < Toes.NumOfCon; k++)
                {
                    LinearExp += Toes.ConPtr[k]->weight * values[count[Toes.ConPtr[k]->LF] + Toes.ConPtr[k]->FromId];
                }
                Term *= DActivation(LinearExp, NN.ActivFunc);
                Term *= differences[count[Froms.ConPtr[n]->LT] + Froms.ConPtr[n]->ToId];
                Term *= Froms.ConPtr[n]->weight;
                weightedSum += Term;
                Term = 1;
            }
            // now differences is holding vectors* with offsets and the goal now is to sum the vectors
            //*     the vector here has dim of neural network number of neurons
            differences[count[i] + j] = weightedSum;
            globaldiff[l * count[NN.NumOfLayers] + count[i] + j] = weightedSum;
        } // finished diffcalc
    }
    __syncthreads();
}
__global__ void preback(neuralnetwork *neuralnetptr, unsigned char **Inputs, unsigned char **Expected, double MLRate, double *globaldiff, double *globalval)
{
    extern __shared__ char s[]; // using char b/c sizeof(char) = 1 byte
    int *count = (int *)s;      // the same meaning as &s[0]
    int j = threadIdx.x;        // indexing neurons
    int l = blockIdx.x;         // indexing inputs and expected
    // values is 1D array where element values[(number of neurons in previous layers to layer i)
    //   + j] is the value of neuron j in layer i using inputs[l]
    // differences is sorted in the same manner for difference values of neurons
    // count will have the size of number of layers of NN and
    //   each element = offset in values and differences for layers.
    // count job is for caching layer offset as it will be used a lot.
    neuralnetwork NN = *neuralnetptr;
    if (j <= NN.NumOfLayers)
    {
        int v = 0;
        for (int i = 0; i < j; i++)
        {
            v += NN.layers[i].NumOfNu;
        }
        count[j] = v;
    } // finished caching
    __syncthreads();
    double *values = (double *)(sizeof(int) * (NN.NumOfLayers + 1) + s);
    double *differences = (double *)(sizeof(double) * count[NN.NumOfLayers] + sizeof(int) * (NN.NumOfLayers + 1) + s);

    if (j < NN.layers[0].NumOfNu)
    {
        values[j] = (double)(Inputs[l][j]) / 128.0;
        globalval[l * count[NN.NumOfLayers] + j] = values[j];
    } // finished InputFirst

    for (int i = 1; i < NN.NumOfLayers; i++)
    {
        __syncthreads();
        if (j < NN.layers[i].NumOfNu)
        {
            NuCon Toes = NN.layers[i].group[j].toes;
            double weightedSum = NN.layers[i].group[j].bias; // initilized with neuron bias
            for (int k = 0; k < Toes.NumOfCon; k++)
            {
                weightedSum += (Toes.ConPtr[k]->weight) * values[count[Toes.ConPtr[k]->LF] + Toes.ConPtr[k]->FromId];
            }
            values[count[i] + j] = Activation(weightedSum, NN.ActivFunc);
            globalval[l * count[NN.NumOfLayers] + count[i] + j] = values[count[i] + j];
        }
    } // finished calc funcitons
    __syncthreads();
    if (j < NN.layers[NN.NumOfLayers - 1].NumOfNu)
    {
        differences[count[NN.NumOfLayers - 1] + j] = MLRate *
                                                     (values[count[NN.NumOfLayers - 1] + j] - Expected[l][j]);
        globaldiff[l * count[NN.NumOfLayers] + count[NN.NumOfLayers - 1] + j] =
            differences[count[NN.NumOfLayers - 1] + j];
    } // finished difflast
    for (int i = NN.NumOfLayers - 2; i > 0; i--)
    {
        __syncthreads();
        if (j < NN.layers[i].NumOfNu)
        {
            double weightedSum = 0;
            NuCon Froms = NN.layers[i].group[j].froms; // the one used for back probagation
            for (int n = 0; n < Froms.NumOfCon; n++)
            {
                double Term = 1;
                double LinearExp = NN.layers[i].group[j].bias;
                NuCon Toes = NN.layers[i].group[j].toes; // the one used for calc
                for (int k = 0; k < Toes.NumOfCon; k++)
                {
                    LinearExp += Toes.ConPtr[k]->weight * values[count[Toes.ConPtr[k]->LF] + Toes.ConPtr[k]->FromId];
                }
                Term *= DActivation(LinearExp, NN.ActivFunc);
                Term *= differences[count[Froms.ConPtr[n]->LT] + Froms.ConPtr[n]->ToId];
                Term *= Froms.ConPtr[n]->weight;
                weightedSum += Term;
                Term = 1;
            }
            // now differences is holding vectors* with offsets and the goal now is to sum the vectors
            //*     the vector here has dim of neural network number of neurons
            differences[count[i] + j] = weightedSum;
            globaldiff[l * count[NN.NumOfLayers] + count[i] + j] = weightedSum;
        } // finished diffcalc
    }
    __syncthreads();
}

// the function goal is to add all the values from globaldiff as vectors and assign them to the network
//<<<N,M>>> where N*M > size(globaldiff)/NumOfInputs (or number of neurons in the neural network)
__global__ void adding(neuralnetwork *neuralnetptr, double *globaldiff, double *globalval, int NumOfInputs,int NumofNeurons)
{
    int j = threadIdx.x + blockDim.x*blockIdx.x;
    // using Sum as intermediate variable to work as cache
    //   instead of using global memory
    if(j<NumofNeurons){
    double Sum = globaldiff[j];
    double Val = globalval[j];
    for (int i = 1; i < NumOfInputs; i++)
    {
        Sum += globaldiff[NumofNeurons * i + j];
        Val += globalval[NumofNeurons * i + j];
    }
    Sum /= NumOfInputs;
    Val /= NumOfInputs;
    int layerIndex = 0, NeuronIndex = j;
    while (NeuronIndex > 0)
    {
        if(neuralnetptr->layers[layerIndex].NumOfNu > NeuronIndex){
            break;
        }
        NeuronIndex -= neuralnetptr->layers[layerIndex].NumOfNu;
        layerIndex += 1;
    }
    neuralnetptr->layers[layerIndex].group[NeuronIndex].difference = Sum;
    neuralnetptr->layers[layerIndex].group[NeuronIndex].value = Val;
    }
    __syncthreads();
}
// the function below utilize the parallelization of gpu by back propagation the network with different inputs
//   however, as a price, it needs more memory since there will be deep copies.
/// @brief this function will evaluate all the differences of neurons from array of inputs and expected results with the supplied count
/// @param neuralnetptr this is neural network pointer
/// @param inputs array of values (array of value to first layer neurons)
/// @param expected array of values (array of expected value from last layer neurons)
/// @param MLRate a parameter that controls speed of changing the hidden parameters (biases and wieghts)
/// @param Count size of inputs and expected (it's needed to protect from segmentation fault)
/// @return nothing, neuralnetptr pointed struct will change
__host__ void PreBackPropagation(neuralnetwork *neuralnetptr, double **inputs, double **expected, double MLRate, int Count)
{
    int sum = neuralnetptr->layers[0].NumOfNu;
    int max = sum;
    for(int i=1;i<neuralnetptr->NumOfLayers;i++){
        sum+= neuralnetptr->layers[i].NumOfNu;
        if(neuralnetptr->layers[i].NumOfNu > max){
            max = neuralnetptr->layers[i].NumOfNu;
        }
    }
    double* globaldiff;
    cudaMalloc((void**)&globaldiff,sizeof(double)*Count*sum);
    preback<<<Count,max,sizeof(double)*sum*2>>>(neuralnetptr,inputs,expected,MLRate,globaldiff);
    cudaDeviceSynchronize();
    adding<<<1,sum>>>(neuralnetptr,globaldiff,Count);
    cudaDeviceSynchronize();
    cudaFree(globaldiff);
}

//<<<N,M>>> where N*M == number of connections
/// to call it, you must call cycle and its formers
__global__ void back(neuralnetwork *neuralnetptr)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x; // indexing wieghts and biases
    neuralnetwork NN = *neuralnetptr;
    if (j < NN.NumOfConnenction)
    { // checking if j in in range of wieghts indeices
        neuron n = NN.layers[NN.connections->LT].group[NN.connections->ToId];
        double diff = n.difference;
        double val = NN.layers[NN.connections->LF].group[NN.connections->FromId].value;

        double LinearExp = n.bias;
        for (int k = 0; k < n.toes.NumOfCon; k++)
        {
            LinearExp += (n.toes.ConPtr[k]->weight) * NN.layers[n.toes.ConPtr[k]->LF].group[n.toes.ConPtr[k]->FromId].value;
        }
        // (del error/del wieght) = (del error/ del finishing neuron)(del finishing neuron /del wiegth)
        //                        = finishing neuron difference * starting neuron value *
        //                          Daf(linear exp of finishing nuron)
        NN.connections[j].weight -= diff * val * DActivation(LinearExp, NN.ActivFunc);
        // (del error/del bias) = (del error/del neuron) * (del neuron/del bias)
        //                      = neuron difference      * Daf(linear exp)
        // however, this command below will be repeated in number of connections that have n
        //                                                               as finishing neuron
        // so to normalize it, it comes the idea to multiply the experssion below by
        //                     1/(num of toes of n)
        NN.layers[NN.connections->LT].group[NN.connections->ToId].bias -= diff * DActivation(LinearExp, NN.ActivFunc) / (n.toes.NumOfCon);
    }
    __syncthreads();
}

__host__ double error(neuralnetwork *neuralnetptr, double *Expected)
{
    double f = 0;
    for (int i = 0; i < neuralnetptr->layers[neuralnetptr->NumOfLayers - 1].NumOfNu; i++)
    {
        f += pow(neuralnetptr->layers[neuralnetptr->NumOfLayers - 1].group[i].value - Expected[i], 2.0);
    }
    return f;
}
__host__ double error(neuralnetwork *neuralnetptr, unsigned char *Expected)
{
    double f = 0;
    double k = 0;
    for (int i = 0; i < neuralnetptr->layers[neuralnetptr->NumOfLayers - 1].NumOfNu; i++)
    {
        printf("the value of neuron %d is %lf and the expected vvalue is %d\n", i, neuralnetptr->layers[neuralnetptr->NumOfLayers - 1].group[i].value, Expected[i]);
        k = neuralnetptr->layers[neuralnetptr->NumOfLayers - 1].group[i].value - Expected[i];
        if ((neuralnetptr->layers[neuralnetptr->NumOfLayers - 1].group[i].value - Expected[i]) < 0)
        {
            k = Expected[i] - neuralnetptr->layers[neuralnetptr->NumOfLayers - 1].group[i].value;
        }
        f += k;
    }
    return f;
}