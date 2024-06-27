// #include "holders.hpp"
#include "mnist.hpp"
// #include "file.hpp"

int main(){
    byte** Image,** Label;
    Image = InputsToNN(Image,"C:\\Users\\Khtably55\\Desktop\\train-images-idx3-ubyte\\train-images.idx3-ubyte",0);
    Label = ExpectedFromNN(Label,"C:\\Users\\Khtably55\\Desktop\\train-images-idx3-ubyte\\train-labels.idx1-ubyte",0);
    free(Image);
    free(Label);
    return 0;
}