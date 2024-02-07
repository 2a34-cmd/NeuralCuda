#ifndef FILE_HPP
#define FILE_HPP
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

#define exists(r) line.find(r) != string::npos

// int NumOfNu(int x);
neuralnetwork FromFile(string fileName);
#endif