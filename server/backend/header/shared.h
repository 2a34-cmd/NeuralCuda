#ifndef SHARED_H
#define SHARED_H
#include <stdint.h>
#include "./file.h"

#define BLOG 6

#define SetBit(X,N)  ((X) |= (1 << (N))) // making Nth bit equal 1 for X
#define ClearBit(X,N)  ((X) &= ~(1 << (N))) // making Nth bit equal 0 for X

typedef struct{
    int ThreadId;
    int ServerFD;
    neuralnetwork* NNptr;
} EnqueuerArguments;

extern uint16_t Working;
#endif