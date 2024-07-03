// #include "holders.hpp"
#include "file.hpp"
#include "mnist.hpp"

int main(){
    // unsigned char** Image,** Label;
    // Image = InputsToNN(Image,"C:\\Users\\Khtably55\\Desktop\\train-images-idx3-ubyte\\t10k-images.idx3-ubyte",0);
    // Label = ExpectedFromNN(Label,"C:\\Users\\Khtably55\\Desktop\\train-images-idx3-ubyte\\t10k-labels.idx1-ubyte",0);
    // free(Image);
    // free(Label);
    neuralnetwork nnP;
    nnP = FromFile("C:/Users/Khtably55/Desktop/ProgrammingProjects(PP)/NeuralNetwork/TestFolder/test3/version10.mn1");
    nnP = FromFile("version10.mn1");
    
    return 0;
}