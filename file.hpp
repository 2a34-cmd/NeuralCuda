#ifndef FILE_HPP
#define FILE_HPP
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

#define exists(r) line.find(r) != string::npos

// int NumOfNu(int x);
void FromFile(string fileName,neuralnetwork* NNp);
void ToFile(string fileName,neuralnetwork* NNp);
#endif