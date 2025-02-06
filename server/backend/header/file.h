#ifndef FILE_H
#define FILE_H
typedef struct{
    // (FromId,LF) is arrow startneuron with index FromId from layer with index LF
    // the same for (ToId,LT) but for arrow finishneuron
    unsigned int FromId,LF,LT,ToId;
    // weight is a multiplier number that has the same job as in wiegthed average or as slope
    double weight;

}connection;

//this struct is to encode list of connections between neurons
//it will be used heavily for forward calculations or back probagation
typedef struct{
    // number of connections in the struct
    unsigned int NumOfCon;
    // pointer (that works as list pointer) to connection pointer
    connection** ConPtr;
}NuCon;

//neuron is the smallest building block for neural network
typedef struct{
    // bias is a number that is added to the initial value and act as y-intercept 
    double bias;
    // it's the value used for nest calculations 
    double value;
    // difference is value that helps adjusting neuron bias and wieghts of connections
    double difference;
    //to keep track of neuron, we need id of it
    unsigned int id;
    // froms are the list of connections that goes from the neuron
    // toes are the list that goes to the neuron
    NuCon froms,toes;
}neuron;


//layer is the next block of neural network
typedef struct{
    //the pointer of neurons and thier count
    neuron* group;
    unsigned int NumOfNu;
    // to keep track of layer
    unsigned int LId;
}layer;


// the enum is encoding a simple int for generic activation functions

typedef enum {
    Tanh=1,
    sigmoid =2,
    ReLU =3,
    Id =4
}ActivationFunc;


//neural network struct keeps track of genral info about the whole network
typedef struct{
    // index of neural network
    int nId;
    // the nonlinear function that is applied before sending the initial value
    ActivationFunc ActivFunc;
    // layer pointer and the count of layers
    layer * layers;
    unsigned int NumOfLayers;
    // connection pointer and thier count
    connection* connections;
    unsigned int NumOfConnenction;
}neuralnetwork;

extern neuralnetwork NN;
#endif