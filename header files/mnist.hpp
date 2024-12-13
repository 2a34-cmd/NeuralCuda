#ifndef MNST_HPP
#define MNIST_HPP
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

//this function gets the unsigned chars from image path starting from startingPos (written as mnist) to be input for neural network  
void InputsToNN(unsigned char*** X,string ImagePath, int startingPos,int finishingPos);


//this function gets the unsigned chars from label path starting from startingPos (written as mnist) to be expected output for neural network backpropagation
void ExpectedFromNN(unsigned char*** X,string LabelPath, int startingPos,int finishingPos);

void InputsToNN(double*** X,string ImagePath, int startingPos,int finishingPos);


//this function gets the unsigned chars from label path starting from startingPos (written as mnist) to be expected output for neural network backpropagation
void ExpectedFromNN(double*** X,string LabelPath, int startingPos,int finishingPos);

#endif