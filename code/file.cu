#include "../header files/file.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
using namespace std;
#define exists(r) line.find(r) != string::npos



//this method extracts real numbers and puts them in a list
//            from the input string (e.g. "1 2.3 -9.4"  =>  {1, 2.3, -9.4})
vector<double> floatExtract(string lin){
    if(lin.find('[') != string::npos){
        int s = lin.size();
    }
    string num;
    vector<double> floats;
    for(int i = 0; i<lin.size();i++){
        if(isdigit(lin[i]) || lin[i]=='.' || lin[i]=='-' || lin[i] == 'E'){
            num += lin[i];
        }else{
            if (!num.empty()){
                floats.push_back(stod(num));
                num.clear();
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
        else if(exists("l")){
            LayerC layer((int)buffer.back());
            NN.layers.push_back(layer);
        }
        else if(exists("nu")){
            NuC nu((int)buffer.front(),buffer.back());
            // neuron nu;
            // nu.bias = buffer.back();
            // nu.id = (int)buffer.front();
            NN.layers.at(NN.layers.size() -1).Neurons.push_back(nu);
        }
        else if(exists("[")){
            connection conn;
            buffer.pop_back();
            conn.weight = buffer.back();
            buffer.pop_back();

            conn.ToId = buffer.back();
            buffer.pop_back();

            conn.LT = buffer.back();
            buffer.pop_back();

            conn.FromId = buffer.back();
            buffer.pop_back();
            
            conn.LF = buffer.back();
            
            NN.cons.push_back(conn);
            NN.layers.at(conn.LT).Neurons.at(conn.ToId).toes++;
            NN.layers.at(conn.LF).Neurons.at(conn.FromId).froms++;
        }
        else if(exists("arr")){
            double num = buffer.at(0);
            if(NN.layers.size() ==1){
                for(int i =0;i< num ;i++){
                    NuC nu(i,0);
                    NN.layers.at(0).Neurons.push_back(nu);
                }
            }
        }
        buffer.clear();
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
    layer* LPtr;
    cudaMallocManaged((void**)&LPtr,sizeof(layer) * n);
    for (int i = 0; i < n; i++)
    {
        layer L;
        L.LId = n-i-1;
        LayerC LC = (LayerC)Input.layers[n-i-1];
        L.NumOfNu = LC.Neurons.size();
        // making pointer of neurons for the layer and allocating memory for it
        neuron* NuPtr;
        cudaMallocManaged((void**)&NuPtr,sizeof(neuron) * L.NumOfNu);
        for (int j = 0; j < L.NumOfNu; j++)
        {
            neuron ne;
            NuC nu = Input.layers[n-i-1].Neurons[j];
            ne.bias = nu.bias;
            ne.difference = 0; ne.value = 0;
            ne.id = j;
            ne.froms.NumOfCon = nu.froms;
            ne.toes.NumOfCon = nu.toes;
            cudaMallocManaged((void**)&ne.froms.ConPtr,ne.froms.NumOfCon * sizeof(connection*));
            cudaMallocManaged((void**)&ne.toes.ConPtr,ne.toes.NumOfCon * sizeof(connection*));
            // cudaMemset(&ne.froms.ConPtr,0,ne.froms.NumOfCon * sizeof(connection*));
            // cudaMemset(&ne.toes.ConPtr,0,ne.toes.NumOfCon * sizeof(connection*));
            NuPtr[j] = ne;
        }
        L.group = NuPtr;

        // nn.layers[n-i-1] = L;
        LPtr[n-i-1] = L;
    }
    nn.layers = LPtr;
    connection* conptr;
    cudaMallocManaged((void**)&conptr,sizeof(connection) * nn.NumOfConnenction);
    for (int i = 0; i < nn.NumOfConnenction; i++)
    {
        conptr[i]= (connection)Input.cons[i];
        neuron FromNeuron = nn.layers[conptr[i].LF].group[conptr[i].FromId];
        neuron ToNeuron = nn.layers[conptr[i].LT].group[conptr[i].ToId];
        for (int a = 0; a < FromNeuron.froms.NumOfCon; a++)
            {
                if(FromNeuron.froms.ConPtr[a] == 0)
                {
                    FromNeuron.froms.ConPtr[a] = &(conptr[i]);
                     break;
                }
            }
            for (int b = 0; b < ToNeuron.toes.NumOfCon; b++)
            {
                if(ToNeuron.toes.ConPtr[b] == 0)
                { 
                    ToNeuron.toes.ConPtr[b] = &(conptr[i]);
                    break;
                }
            }
    }
    nn.connections = conptr;
    return nn;
}


//this method is just the combination of the previous 2 methods
//it takes the file path in return of pointer towards neural network struct
neuralnetwork FromFile(string fileName){
    neuralnetwork NN = Converter(file(fileName));
    return NN;
}

// you could say the way file written makes it easier to write than to read
//  writing process doesn't need conversion to class or using floatExtract method


//this method writes to file with fileName path with the information given
//    by struct pointed by NNp
void ToFile(string fileName,neuralnetwork* NNp){
    std::string line;
    ofstream Writer;
    Writer.open(fileName);
    if(!Writer.is_open()){
        cout<< "there are problems, output file stream isn't working";
        return;
    }
    Writer << "s\n";
    
    line = "net" + to_string(NNp->nId) + ":" + to_string(static_cast<int>(NNp->ActivFunc));
    
    Writer << line << ";\n";


    Writer << "l0;\narr" << to_string(NNp->layers[0].NumOfNu) << ";\n";
    for(int i=1;i<(NNp->NumOfLayers);i++){
        line = "l" + to_string(i);
        Writer << line << ";\n";
        for(int j =0;j<(NNp->layers[i].NumOfNu);j++){
            line = "nu " + to_string(j) + " : " + to_string(NNp->layers[i].group[j].bias);
            Writer << line << ";\n";
        }
    }

    Writer << "c\n";
    for(int i =0; i< (NNp->NumOfConnenction); i++){
        connection con = NNp->connections[i];
        line = "[" + to_string(con.LF) + "," + to_string(con.FromId) + "][" +
            to_string(con.LT) + "," + to_string(con.ToId) + "]" + to_string(con.weight);
        Writer << line << ":0;\n";
    }

    Writer.flush();
    Writer.close();
    return;
}