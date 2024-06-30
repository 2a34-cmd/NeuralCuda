#ifndef MNST_HPP
#define MNIST_HPP
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

//this function gets the unsigned chars from image path starting from startingPos (written as mnist) to be input for neural network  
unsigned char** InputsToNN(unsigned char** X,string ImagePath, int startingPos);


//this function gets the unsigned chars from label path starting from startingPos (written as mnist) to be expected output for neural network backpropagation
unsigned char** ExpectedFromNN(unsigned char** X,string LabelPath, int startingPos);

#endif