#ifndef HOLDERS_HPP
#define HOLDERS_HPP
#include <vector>


//even if we can use classes, we need to use structs for moving data and results between cpu and gpu
//
//neural network consists of layers which are connected like directed graphs
//layers consists of neurons

//connection is directed arrow from some neuron to another
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
    // froms are the list of connections that goes to the neuron
    // toes are the list that goes from the neuron
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

// the enum is encoding a simple int for generic activation functions
typedef enum{
    Tanh=1,sigmoid,ReLU,Id
}ActivationFunc;

// the classes below are used for getting information from files and to them
//     they aren't used in the real computation b/c in gpu proccessing,
//     we need to use pointers in the first place

//      classes has the property of pre-initializing that doesn't exist in structs
//      structs need to be fully initialized to work in the first place


class NueralNet
{
public:
    int nId;
    ActivationFunc ActivationFunction;
    std::vector<LayerC> layers;
    std::vector<connection> cons;
    NueralNet(int Id,int AF);
    NueralNet();
    ~NueralNet();
};
class LayerC
{
private:
    int LId;
public:
    std::vector<NuC> Neurons;
    LayerC(int LId);
    ~LayerC();
};
class NuC
{
public:
    double bias,value,difference;
    unsigned int id;
    int froms,toes;
    NuC(int id,double bias);
    ~NuC();
};
#endif