#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include "holders.hpp"
using namespace std;

#define exists(r) line.find(r) != string::npos



//this method extracts real numbers and puts them in a list
//            from the input string (e.g. "1 2.3 -9.4"  =>  {1, 2.3, -9.4})
vector<double> floatExtract(string lin){
    string num;
    vector<double> floats;
    for(int i = 0; i<lin.size();i++){
        if(isdigit(lin[i]) || lin[i]=='.' || lin[i]=='-'){
            num += lin[i];
        }else{
            if (!num.empty()){
                floats.push_back(stod(num));
            }
        }
    }
    return floats;
}

// this method translates the file with path fN to member of class NeuralNet
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
    File.close();
    return NN;
}

//I know I could add information about number of layers and neurons to skip 
//      the converting between class and struct, but it's a design deccision


//this method converts class member into struct
neuralnetwork Converter(NueralNet Input){
    neuralnetwork nn;

    nn.ActivFunc = Input.ActivationFunction;
    nn.nId = Input.nId;
    nn.NumOfConnenction = Input.cons.size();
    nn.NumOfLayers = Input.layers.size();

    int n = nn.NumOfLayers;
    // making pointer of layer list and allocating the size
    // this is 1 reason from many why I can't translate file to neuralnetwork struct directly 
    layer* LPtr = (layer*) malloc(sizeof(layer) * n);
    for (int i = 0; i < n; i++)
    {
        layer L;
        L.LId = n-i-1;
        LayerC LC = (LayerC)Input.layers.back();
        L.NumOfNu = LC.Neurons.size();
        // making pointer of neurons for the layer and allocating memory for it
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


//this method is just the combination of the previous 2 methods
//it takes the file path in return of pointer towards neural network struct
void FromFile(string fileName,neuralnetwork* NNp){
    neuralnetwork NN = Converter(file(fileName));
    NNp  = &NN;
}

// you could say the way file written makes it easier to write than to read
//  writing process doesn't need conversion to class or using floatExtract method


//this method writes to file with fileName path with the information given
//    by struct pointed by NNp
void ToFile(string fileName,neuralnetwork* NNp){
    std::string line;
    ofstream Writer(fileName);

    Writer << "s\n";;
    line = "net" + to_string(NNp->nId) + ":" + to_string(static_cast<int>(NNp->ActivFunc));
    Writer << line << ";\n";
    for(int i=0;i<(NNp->NumOfLayers);i++){
        line = "l" + to_string(i);
        Writer << line << ";\n";;
        for(int j =0;j<(NNp->layers[i].NumOfNu);j++){
            line = "nu" + to_string(j) + ": " + to_string(NNp->layers[i].group[j].bias);
            Writer << line << ";\n";
        }
    }

    Writer << "c\n";;
    for(int i =0; i< (NNp->NumOfConnenction); i++){
        connection con = NNp->connections[i];
        line = "[" + to_string(con.LF) + "," + to_string(con.FromId) + "][" +
            to_string(con.LT) + "," + to_string(con.ToId) + "]" + to_string(con.weight);
        Writer << line << ";\n";;
    }
    Writer.close();
    return;
}