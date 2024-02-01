#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include "holders.cpp"
using namespace std;

#define exists(r) line.find(r) != string::npos
//int NumOfNu(int x){
//     switch (x)
//     {
//     case 0:
//         return 784;
//     case 1:
//         return 257;
//     case 2:
//         return 84;
//     case 3:
//         return 29;
//     case 4:
//         return 10;
//     default:
//         return NAN;
//     }
// }




vector<double> floatExtract(string lin){
    string num;
    vector<double> floats;
    for(int i = 0; i<lin.size();i++){
        if(isdigit(lin[i]) || lin[i]=='.'){
            num += lin[i];
        }else{
            if (!num.empty()){
                floats.push_back(stod(num));
            }
        }
    }
    return floats;
}

NueralNet file(string fN){
    string line;
    class NueralNet NN;

    ifstream File(fN);
    vector<double> buffer;

    while (!File.eof())
    {
        getline(File,line);
        buffer = floatExtract(line);
        if(exists("net")){
            NN.nId = (int) buffer.front();
            NN.ActivationFunction = (ActivationFunc)(int) buffer.back();
        }
        if(exists("l")){
            LayerC layer((int)buffer.back());
            NN.layers.push_back(layer);
        }
        if(exists("nu")){
            NuC nu((int)buffer.front(),buffer.back());
            // neuron nu;
            // nu.bias = buffer.back();
            // nu.id = (int)buffer.front();
            LayerC layer = (LayerC)NN.layers.back();
            layer.Neurons.push_back(nu);
        }
        if(exists("[")){
            connection conn;

            conn.weight = buffer.back();
            buffer.pop_back();

            conn.LT = buffer.back();
            buffer.pop_back();

            conn.ToId = buffer.back();
            buffer.pop_back();

            conn.LF = buffer.back();
            buffer.pop_back();
            
            conn.FromId = buffer.back();
            
            NN.cons.push_back(conn);
            NN.layers[conn.LT].Neurons[conn.ToId].toes++;
            NN.layers[conn.LF].Neurons[conn.FromId].froms++;
        }
        buffer.empty();
    }
    
    return NN;
}
neuralnetwork Converter(NueralNet Input){
    neuralnetwork nn;

    nn.ActivFunc = Input.ActivationFunction;
    nn.nId = Input.nId;
    nn.NumOfConnenction = Input.cons.size();
    nn.NumOfLayers = Input.layers.size();

    layer* LPtr = (layer*) malloc(sizeof(layer) * nn.NumOfLayers);
    int n = nn.NumOfLayers;
    for (int i = 0; i < nn.NumOfLayers; i++)
    {
        layer L;
        L.LId = n-i-1;
        LayerC LC = (LayerC)Input.layers.back();
        L.NumOfNu = LC.Neurons.size();

        neuron* NuPtr = (neuron*) malloc(sizeof(neuron) * L.NumOfNu);
        for (int j = 0; j < L.NumOfNu; j++)
        {
            neuron ne;
            NuC nu = Input.layers[n-i-1].Neurons[j];
            ne.bias = nu.bias;
            ne.difference = 0; ne.value = 0;
            ne.id = j;
            ne.froms.NumOfCon = nu.froms;
            ne.toes.NumOfCon = nu.toes;
            ne.froms.ConPtr = (connection**)malloc(sizeof(connection*) * ne.froms.NumOfCon);
            
            
            
            ne.toes.ConPtr = (connection**) malloc(sizeof(connection*) * ne.toes.NumOfCon);
            NuPtr[j] = ne;
        }
        L.group = NuPtr;

        nn.layers[n-i-1] = L;
    }
    nn.layers = LPtr;
    int m = 0;
    connection* conptr=(connection*)malloc(sizeof(connection) * nn.NumOfConnenction);
    for (int i = 0; i < nn.NumOfConnenction; i++)
    {
        conptr[i]= (connection)Input.cons[i];
        neuron ne = nn.layers[conptr[i].LF].group[conptr[i].FromId];
        for (int a = 0; a < ne.froms.NumOfCon; a++)
            {
                if(ne.froms.ConPtr[a] == 0){ne.froms.ConPtr[a] = &(conptr[i]); break;}
            }
            for (int b = 0; b < ne.toes.NumOfCon; b++)
            {
                if(ne.toes.ConPtr[b] == 0){ ne.toes.ConPtr[b] = &(conptr[i]); break;}
            }
    }
    nn.connections = conptr;
}

void FromFile(string fileName,neuralnetwork* NNp){
    neuralnetwork NN = Converter(file(fileName));
    NNp  = &NN;
}