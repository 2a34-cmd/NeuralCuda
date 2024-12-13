#ifndef FILE_HPP
#define FILE_HPP
using namespace std;
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "holders.hpp"

#define exists(r) line.find(r) != string::npos

// int NumOfNu(int x);
neuralnetwork FromFile(string fileName);
void ToFile(string fileName,neuralnetwork* NNp);
#endif