#include <vector>
struct neuron{
    double bias;
    double value;
    double difference;
    unsigned int id;
    NuCon froms,toes;
};
struct layer{
    struct neuron* group;
    unsigned int NumOfNu;
    unsigned int LId;
};
struct connection{
    unsigned int FromId,LF,LT,ToId;
    double weight;

};
struct NuCon
{
    unsigned int NumOfCon;
    connection** ConPtr;
};

struct neuralnetwork{
    int nId;
    enum ActivationFunc ActivFunc;
    struct layer * layers;
    unsigned int NumOfLayers;
    struct connection* connections;
    unsigned int NumOfConnenction;
};

// enum numtype{
//     bias,weight,neuronId,layerId,networkId,networkFunc,CNFId,CLFId,CNTId,CLTId
// };
enum ActivationFunc{
    Tanh=1,sigmoid,ReLU,Id
};


class NueralNet
{
public:
    int nId;
    ActivationFunc ActivationFunction;
    vector<LayerC> layers;
    vector<connection> cons;
    NueralNet(int Id,int AF);
    NueralNet();
    ~NueralNet();
};
NueralNet::NueralNet(int Id,int AF)
{
    nId = Id;
    ActivationFunction = (ActivationFunc)AF; 
}
NueralNet::NueralNet()
{
}
NueralNet::~NueralNet()
{
}

class LayerC
{
private:
    int LId;
public:
    vector<NuC> Neurons;
    LayerC(int LId);
    ~LayerC();
};
LayerC::LayerC(int Id)
{
    LId = Id;
}
LayerC::~LayerC()
{
}

class NuC
{
public:
    double bias,value,difference;
    unsigned int id;
    int froms,toes;
    NuC(int id,double bias);
    ~NuC();
};
NuC::NuC(int id,double bias)
{
    difference = 0;
    value = 0;
    bias = bias;
    id = id;
    froms = 1;
    toes=1;
}

NuC::~NuC()
{
}
