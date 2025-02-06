#define _GNU_SOURCE
#include <pthread.h>
#include <sys/socket.h>
#include <string.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <ctype.h>
#include "../header/calc.h"

// there are 2 thread types: enqueuer, and handler
//  enqueuer jobs
//  0-initiliziation: make the cache (html,css,js,neuralnetwork)
//  1-accept client request and read the request
//  handler jobs
//  2-parse what is needed (get html,css,js,etc.)
//  3-read the cache or use neuralnetwork
//  (possible)4- chunking and sending the data
//  5-close the connection

void enQueue(int value)
{
    if (queue.rear == 3 - 1)
        printf("\nQueue is Full!!");
    else
    {
        if (queue.front == -1)
            queue.front = 0;
        queue.rear++;
        queue.items[queue.rear] = value;
    }
}
int deQueue()
{
    if (queue.front == -1)
    {
        return -1;
    }
    else
    {
        int temp = queue.front;
        queue.front++;
        if (queue.front > queue.rear)
            queue.front = queue.rear = -1;
        return queue.items[temp];
    }
}
void intExtract(const char *lin, unsigned char *chars, ssize_t linlength)
{
    if(linlength == -1){
        return;
    }
    unsigned char num[5] = {0};
    int k = 0, l = 0, Doub = 0;
    for (int i = 0; i < linlength; i++)
    {
        if (isdigit(lin[i]))
        {
            num[l] = lin[i];
            l++;
        }
        else
        {
            Doub = strnlen(num,5);
            if (Doub != 0)
            {
                sscanf(num, "%hhu", &(chars[k]));
                memset(num, 0, 5 * sizeof(unsigned char));
                k++;
                l = 0;
            }
        }
    }
    return;
}

int HandleAsync(int ClientFD)
{
    char buffer[4096] = {0};
    int error = 0;
    error = recv(ClientFD, buffer, 4096 * sizeof(char), 0);
    if (error == -1)
    {
        perror("recv");
        return -1;
    }
    printf("client %d has request\n%s\n", ClientFD, buffer);
    if (strstr(buffer, "GET") == buffer)
    {
        // getting file name
        char *NewPtr = strstr(buffer, " HTTP");
        char *FileString = (char *)calloc((NewPtr - &buffer[4]) + 2*sizeof(char), sizeof(char));
        memcpy(&FileString[1], &buffer[4], NewPtr - &buffer[4]);
        FileString[0] = '.';
        // checking file type
        if (strlen(FileString) == 2)
        {
            pthread_rwlock_rdlock(&HTMLCacheLock);
            int Diff = strnlen(HTML, 8191);
            send(ClientFD, HTML, Diff * sizeof(char), 0);
            pthread_rwlock_unlock(&HTMLCacheLock);
            close(ClientFD);
        }
        else if (strstr(FileString, "css") != NULL)
        {
            pthread_rwlock_rdlock(&CSSCacheLock);
            int Diff = strnlen(CSS,8191);
            send(ClientFD, CSS, Diff * sizeof(char), 0);
            pthread_rwlock_unlock(&CSSCacheLock);
            close(ClientFD);
        }
        else if (strstr(FileString, "js") != NULL)
        {
            pthread_rwlock_rdlock(&JSCacheLock);
            int Diff = strnlen(JS,8191);
            send(ClientFD, JS, Diff * sizeof(char), 0);
            pthread_rwlock_unlock(&JSCacheLock);
            close(ClientFD);
        }
        else if (strstr(FileString, "/results.data/") != NULL)
        {
            // using neural network
            unsigned char Inputs[784] = {0};
            double* Outputs = malloc(10*sizeof(double));
            intExtract(buffer,Inputs,784);
            Outputs = CalcNeuralNetworkNoSE(Inputs);
            memset(buffer, 0, 4096 * sizeof(char));
            char temp[15] = {0};
            strcat(buffer,"HTTP/1.1 200 OK\r\nContent-Type: text/json\r\n\r\n");
            for(int j=0;j<10;j++){
                sprintf(temp,"%.2lf,",Outputs[j]);
                strcat(buffer,temp);
            }
            int Diff = strnlen(buffer,8191);

            send(ClientFD, buffer, Diff * sizeof(char), 0);
            free(Outputs);
            close(ClientFD);
        }
        else
        {
            // sending error
            send(ClientFD, "HTTP/1.1 418 I'm a teapot\r\nContent-Type: text plain\r\n\r\nYou shouldn't request from a teapot.", 92 * sizeof(char), 0);
            close(ClientFD);
        }
        free(FileString);
        return 0;
    }
    else
    {
        // client want method other than GET, server doesn't
        send(ClientFD, "HTTP/1.1 405 Method Not Allowed\r\nContent-Type: text plain\r\n\r\n405 Method Not Allowed", 84 * sizeof(char), 0);
        close(ClientFD);
        return 0;
    }
    perror("shouldn't be here\n");
    return -2;
}

void *Handler(void *args)
{
    int clientFD = -2;
    HandlerArguments ArgStruct = *((HandlerArguments *)args);
    while (ReadBit(Working, ArgStruct))
    {
        pthread_mutex_lock(&QueueLock);
        clientFD = deQueue();
        pthread_mutex_unlock(&QueueLock);
        if (clientFD == -2)
        {
            perror("handler thread shouldn't be here\n");
            return NULL;
        }else if (clientFD == -1)
        {
            sleep(1);
        }
        else
        {
            int error = HandleAsync(clientFD);
            if(error <0){
                return NULL;
            }
        }
    }
    return NULL;
}
void *Enqueuer(void *args)
{
    // initilize
    EnqueuerArguments ArgStruct = *((EnqueuerArguments *)args);
    int HandlerIds[BLOG] = {0};
    int HtmlFile = open("../../frontend/index.html", 'r');
    if (HtmlFile == -1)
    {
        printf("HTML file doesn't open\n");
        return NULL;
    }
    pthread_rwlock_wrlock(&HTMLCacheLock);
    read(HtmlFile, &HTML[44], (8192 - 44) * sizeof(char));
    strncpy(HTML, "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n", 44 * sizeof(char));
    pthread_rwlock_unlock(&HTMLCacheLock);
    close(HtmlFile);
    int CssFile = open("../../frontend/style.css", 'r');
    if (CssFile == -1)
    {
        printf("CSS file doesn't open\n");
        return NULL;
    }
    pthread_rwlock_wrlock(&CSSCacheLock);
    read(CssFile, &CSS[43], (8192 - 43) * sizeof(char));
    strncpy(CSS, "HTTP/1.1 200 OK\r\nContent-Type: text/css\r\n\r\n", 43 * sizeof(char));
    pthread_rwlock_unlock(&CSSCacheLock);
    close(CssFile);
    int JsFile = open("../../frontend/app.js", 'r');
    if (JsFile == -1)
    {
        printf("JS file doesn't open\n");
        return NULL;
    }
    pthread_rwlock_wrlock(&JSCacheLock);
    read(JsFile, &JS[42], (8192 - 42) * sizeof(char));
    strncpy(JS, "HTTP/1.1 200 OK\r\nContent-Type: text/js\r\n\r\n", 42 * sizeof(char));
    pthread_rwlock_unlock(&JSCacheLock);
    close(JsFile);
    for (int i = 0; i < BLOG; i++)
    {
        HandlerArgs[i] = i + 1;
        SetBit(Working,i+1);
        HandlerIds[i] = pthread_create(&threads[i], NULL, Handler, &HandlerArgs[i]);
    }
    // enqueue
    while (ReadBit(Working, ArgStruct.ThreadId))
    {
        int ClientFD = accept(ArgStruct.ServerFD, NULL, NULL);
        if (ClientFD == -1)
        {
            perror("accept");
            for (int i = 0; i < BLOG; i++)
            {
                ClearBit(Working,i+1);
                pthread_join(HandlerIds[i], NULL);
            }
            return NULL;
        }
        else
        {
            pthread_mutex_lock(&QueueLock);
            enQueue(ClientFD);
            pthread_mutex_unlock(&QueueLock);
        }
    }
    // tidy up
    for (int i = 0; i < BLOG; i++)
    {
        ClearBit(Working,i+1);
        pthread_join(HandlerIds[i], NULL);
    }
    return NULL;
}


int Handle(int ServerFD)
{
    char buffer[8192];
    int error = 0;
    // accepting
    int ClientFD = accept(ServerFD, NULL, NULL);
    if (ClientFD == -1)
    {
        perror("accept");
        return -4;
    }
    // receaving
    memset(buffer, 0, 8192 * sizeof(char));
    error = recv(ClientFD, buffer, 2048 * sizeof(char), 0);
    if (error == -1)
    {
        perror("recv");
        return -5;
    }
    printf("client %d has request\n%s\n", ClientFD, buffer);

    // checking HTTP methods
    if (strstr(buffer, "GET") == buffer)
    {
        // getting file name
        char *NewPtr = strstr(buffer, " HTTP");
        char *FileString = (char *)calloc((NewPtr - &buffer[4]) + sizeof(char), sizeof(char));
        memcpy(FileString + 1, &buffer[4], NewPtr - &buffer[4]);
        FileString[0] = '.';

        // checking file type
        if (strlen(FileString) == 2)
        {

            // sending html
            int HTML = open("./index.html", 'r');
            if (HTML == -1)
            {
                send(ClientFD, "HTTP/1.1 404 Not Found\r\nContent-Type: text plain\r\n\r\n404 Not Found", 66 * sizeof(char), 0);
                close(ClientFD);
                return -10;
            }
            memset(buffer, 0, 8192 * sizeof(char));
            read(HTML, &buffer[44], (8192 - 44) * sizeof(char));
            strncpy(buffer, "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n", 44 * sizeof(char));

            buffer[8191] = '\0'; // to ensure strlen don't get off bounds
            int Diff = strlen(buffer);
            send(ClientFD, buffer, Diff * sizeof(char), 0);
            close(ClientFD);
            return 0;
        }
        else if (strstr(FileString, "css") != NULL)
        {
            // sending css
            int CSS = open(FileString, 'r');
            if (CSS == -1)
            {
                send(ClientFD, "HTTP/1.1 404 Not Found\r\nContent-Type: text plain\r\n\r\n404 Not Found", 66 * sizeof(char), 0);
                close(ClientFD);
                return -10;
            }
            memset(buffer, 0, 8192 * sizeof(char));
            read(CSS, &buffer[43], (8192 - 43) * sizeof(char));
            strncpy(buffer, "HTTP/1.1 200 OK\r\nContent-Type: text/css\r\n\r\n", 43 * sizeof(char));

            buffer[8191] = '\0'; // to ensure strlen don't get off bounds
            int Diff = strlen(buffer);

            send(ClientFD, buffer, Diff * sizeof(char), 0);
            close(ClientFD);
            return 0;
        }
        else if (strstr(FileString, "js") != NULL)
        {
            // sending js
            int JS = open(FileString, 'r');
            if (JS == -1)
            {
                send(ClientFD, "HTTP/1.1 404 Not Found\r\nContent-Type: text plain\r\n\r\n404 Not Found", 66 * sizeof(char), 0);
                close(ClientFD);
                return -10;
            }
            memset(buffer, 0, 8192 * sizeof(char));
            read(JS, &buffer[50], (8192 - 50) * sizeof(char));
            strncpy(buffer, "HTTP/1.1 200 OK\r\nContent-Type: text/javascript\r\n\r\n", 50 * sizeof(char));

            buffer[8191] = '\0'; // to ensure strlen don't get off bounds
            int Diff = strlen(buffer);

            send(ClientFD, buffer, Diff * sizeof(char), 0);
            close(ClientFD);
            return 0;
        }
        // else if (strstr(FileString, "results.data") != NULL)
        // {
        //     // using neural network

        //     CalcNeuralNetwork(NN, NULL, NULL);

        //     memset(buffer, 0, 8192 * sizeof(char));
        //     buffer[8191] = '\0'; // to ensure strlen don't get off bounds
        //     int Diff = strlen(buffer);

        //     send(ClientFD, buffer, Diff * sizeof(char), 0);
        //     close(ClientFD);
        // }
        else
        {
            // sending error
            send(ClientFD, "HTTP/1.1 418 I'm a teapot\r\n\r\nYou shouldn't request from a teapot.", 66 * sizeof(char), 0);
            close(ClientFD);
            return 0;
        }
        free(FileString);
    }
    else
    {
        // client want method other than GET, server doesn't
        send(ClientFD, "HTTP/1.1 405 Method Not Allowed\r\nContent-Type: text plain\r\n\r\n405 Method Not Allowed", 84 * sizeof(char), 0);
        close(ClientFD);
        return 0;
    }
    return 0;
}
