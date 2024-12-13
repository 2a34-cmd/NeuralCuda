#include "../header files/holders.hpp"
using namespace std;





// typedef struct{
//     unsigned int FromId,LF,LT,ToId;
//     double weight;
// }connection;


// typedef struct{
//     unsigned int NumOfCon;
//     connection** ConPtr;
// }NuCon;

// typedef struct{
//     double bias;
//     double value;
//     double difference;
//     unsigned int id;
//     NuCon froms,toes;
// }neuron;


// typedef struct{
//     neuron* group;
//     unsigned int NumOfNu;
//     unsigned int LId;
// }layer;


// typedef enum{
//     Tanh=1,sigmoid =2,ReLU =3,Id =4
// }ActivationFunc;


// typedef struct{
//     int nId;
//     ActivationFunc ActivFunc;
//     layer * layers;
//     unsigned int NumOfLayers;
//     connection* connections;
//     unsigned int NumOfConnenction;
// }neuralnetwork;




// class NuC
// {
// public:
//     double bias,value,difference;
//     unsigned int id;
//     int froms,toes;
//     NuC(int id,double bias);
//     ~NuC();
// };
NuC::NuC(int id,double bias)
{
    difference = 0;
    value = 0;
    NuC::bias = bias;
    NuC::id = id;
    froms = 0;
    toes=0;
}
NuC::~NuC()
{
}

// class LayerC
// {
// private:
//     int LId;
// public:
//     vector<NuC> Neurons;
//     LayerC(int LId);
//     ~LayerC();
// };
LayerC::LayerC(int Id)
{
    LId = Id;
}
LayerC::~LayerC()
{

}



// class NueralNet
// {
// public:
//     int nId;
//     ActivationFunc ActivationFunction;
//     vector<LayerC> layers;
//     vector<connection> cons;
//     NueralNet(int Id,int AF);
//     NueralNet();
//     ~NueralNet();
// };
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


