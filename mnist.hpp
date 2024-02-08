#ifndef MNST_HPP
#define MNIST_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

//this function gets the bytes from image path starting from startingPos (written as mnist) to be input for neural network  
byte** InputsToNN(string ImagePath, int startingPos);


//this function gets the bytes from label path starting from startingPos (written as mnist) to be expected output for neural network backpropagation
byte** ExpectedFromNN(string LabelPath, int startingPos);

#endif