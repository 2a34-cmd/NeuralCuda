#ifndef HOLDERS_HPP
#define HOLDERS_HPP
#include <vector>

typedef struct{
    unsigned int FromId,LF,LT,ToId;
    double weight;

}connection;
typedef struct{
    unsigned int NumOfCon;
    connection** ConPtr;
}NuCon;
typedef struct{
    double bias;
    double value;
    double difference;
    unsigned int id;
    NuCon froms,toes;
}neuron;
typedef struct{
    neuron* group;
    unsigned int NumOfNu;
    unsigned int LId;
}layer;
typedef struct{
    int nId;
    ActivationFunc ActivFunc;
    layer * layers;
    unsigned int NumOfLayers;
    connection* connections;
    unsigned int NumOfConnenction;
}neuralnetwork;

typedef enum{
    Tanh=1,sigmoid,ReLU,Id
}ActivationFunc;


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