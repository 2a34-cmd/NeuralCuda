#include "./shared.h"

#define FileCacheSize 8192

#define ReadBit(X,N) (((X)>>(N))& 1) // reading Nth bit of X


void CalcNeuralNetwork(unsigned char *Inputs,double* Outputs);
double* CalcNeuralNetworkNoSE(unsigned char *Inputs);

typedef int HandleArgs;

typedef int HandlerArguments;

typedef struct{
    int items[BLOG];
    int front,rear;
}Queue;


Queue queue = { 
    .items = {0},
    .front = -1,
    .rear = -1
};
char HTML[FileCacheSize] = {0},CSS[FileCacheSize] = {0},JS[FileCacheSize] = {0};
pthread_t threads[BLOG];
HandleArgs handleargs[BLOG] = {0};
HandlerArguments HandlerArgs[BLOG] = {0};
pthread_mutex_t QueueLock = PTHREAD_MUTEX_INITIALIZER;
pthread_rwlock_t HTMLCacheLock = PTHREAD_RWLOCK_INITIALIZER;
pthread_rwlock_t CSSCacheLock = PTHREAD_RWLOCK_INITIALIZER;
pthread_rwlock_t JSCacheLock = PTHREAD_RWLOCK_INITIALIZER;
