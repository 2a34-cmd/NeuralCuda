#include "../header/file.h"
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <fcntl.h>
#define exists(r) strstr(line, r) != NULL

neuralnetwork NN;

// this method extracts real numbers and puts them in a list
//             from the input string (e.g. "1 2.3 -9.4"  =>  {1, 2.3, -9.4})
void floatExtract(const char *lin, double *floats, ssize_t linlength)
{
    if(linlength == -1){
        return;
    }
    char num[15];
    memset(num,0,15*sizeof(char));
    int k = 0, l = 0, Doub = 0;
    for (int i = 0; i < linlength; i++)
    {
        if (isdigit(lin[i]) | lin[i] == '.' | lin[i] == '-' | lin[i] == 'E')
        {
            num[l] = lin[i];
            l++;
        }
        else
        {
            Doub = strnlen(num,15);
            if (Doub != 0)
            {
                sscanf(num, "%lf", &(floats[k]));
                memset(num, 0, 15 * sizeof(char));
                k++;
                l = 0;
            }
        }
    }
    return;
}

// this method translates the file with path fN to member of class NeuralNet
void FromFile(char* fN)
{
    char* line = calloc(40,sizeof(char));
    ssize_t LineLength = 0;
    size_t useless = 0;
    FILE* fileS = fopen(fN, "r");

    int numofL = 0, numofC = 0;
    if (fileS == NULL)
    {
        printf("there are problems in locating file\n");
        return;
    }
    double buffer[8] = {0};
    LineLength = getline((char**)&line, &useless, fileS);
    while ((LineLength = getline((char**)&line, &useless, fileS)) != -1)
    {
        floatExtract(line, buffer, LineLength);
        if (exists("net"))
        {
            // this line in the shape
            //  "net(network id):(activatin function number):(number of layers):(number of connections);"
            NN.nId = (int)buffer[0];
            NN.ActivFunc = (ActivationFunc)(int)buffer[1];
            NN.NumOfLayers = (int)buffer[2];
            NN.NumOfConnenction = (int)buffer[3];
            NN.layers = calloc(NN.NumOfLayers, sizeof(layer));
            NN.connections = calloc(NN.NumOfConnenction, sizeof(connection));
        }
        else if (exists("l"))
        {
            //"l(layer id):(number of neurons);"
            NN.layers[(int)buffer[0]].LId = (int)buffer[0];
            NN.layers[(int)buffer[0]].NumOfNu = (int)buffer[1];
            NN.layers[(int)buffer[0]].group = calloc(NN.layers[(int)buffer[0]].NumOfNu, sizeof(neuron));
            numofL++;
        }
        else if (exists("nu"))
        {
            //"nu (neuron id) : (bias) : (number of toes connections) : (number of froms connections);"
            //"toes are the ones used in calculations, froms are used in back propapegations"
            //"toes are connections going to this nueron, froms are going from this nueron"
            NN.layers[numofL - 1].group[(int)buffer[0]].id = buffer[0];
            NN.layers[numofL - 1].group[(int)buffer[0]].bias = buffer[1];
            NN.layers[numofL - 1].group[(int)buffer[0]].difference = 0;
            NN.layers[numofL - 1].group[(int)buffer[0]].value = 0;
            NN.layers[numofL - 1].group[(int)buffer[0]].toes.NumOfCon = (int)buffer[2];
            NN.layers[numofL - 1].group[(int)buffer[0]].froms.NumOfCon = (int)buffer[3];
            NN.layers[numofL - 1].group[(int)buffer[0]].toes.ConPtr = calloc(NN.layers[numofL - 1].group[(int)buffer[0]].toes.NumOfCon, sizeof(connection*));
            NN.layers[numofL - 1].group[(int)buffer[0]].froms.ConPtr = calloc(NN.layers[numofL - 1].group[(int)buffer[0]].froms.NumOfCon, sizeof(connection*));
        }
        else if (exists("["))
        {
            // connction is from (Fromnueron,Fromlayer) to (Tonueron,Tolayer) witth weight
            //"[(Fromlayer):(Fromnueron)][(Tolayer):(Tonueron)]: weight:"
            NN.connections[numofC].LF = (int)buffer[0];
            NN.connections[numofC].FromId = (int)buffer[1];
            NN.connections[numofC].LT = (int)buffer[2];
            NN.connections[numofC].ToId = (int)buffer[3];
            NN.connections[numofC].weight = buffer[4];
            NN.layers[NN.connections[numofC].LT].group[NN.connections[numofC].ToId].toes.ConPtr[NN.connections[numofC].FromId] = &NN.connections[numofC];
            NN.layers[NN.connections[numofC].LF].group[NN.connections[numofC].FromId].froms.ConPtr[NN.connections[numofC].ToId] = &NN.connections[numofC];
            numofC++;
        }
        else if (exists("arr"))
        {
            double num = buffer[0];
            for (int i = 0; i < num; i++)
            {
                neuron nu = NN.layers[numofL -1].group[i];
                NN.layers[numofL -1].group[i].id = 0;
                NN.layers[numofL -1].group[i].bias = 0;
                NN.layers[numofL -1].group[i].value = 0;
                NN.layers[numofL -1].group[i].difference = 0;
                NN.layers[numofL -1].group[i].toes.NumOfCon = (int)buffer[1];
                NN.layers[numofL -1].group[i].froms.NumOfCon = (int)buffer[2];
                NN.layers[numofL -1].group[i].toes.ConPtr = calloc(NN.layers[numofL -1].group[i].toes.NumOfCon,sizeof(connection));
                NN.layers[numofL -1].group[i].froms.ConPtr = calloc(NN.layers[numofL -1].group[i].froms.NumOfCon,sizeof(connection));
            }
        }
        memset(buffer, 0, 8 * sizeof(double));
    }
    fclose(fileS);
    free(line);
    return;
}
