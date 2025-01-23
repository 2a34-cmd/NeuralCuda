#ifndef HANDLE_C
#include <pthread.h>
#include <sys/socket.h>
#include <string.h>
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#define HANDLE_C
#endif


// the commened segment below will be the backbone of async server
//  for the next commit


// 1-accept client request and read the request
// 2-parse what is needed (get html,css,js,etc.)
// 3-read the needed file
// 4-send buffers of size 1024 of the files (chunking)
// 5-close the connection

// typedef void (*pcb)(int ClientFD); // pointer to void ___ (int a)
// typedef struct parameter
// {
//     int a;
//     pcb callback;
// } parameter;

// void *callback_thread(void *p1) // get p1 and then use p1->callback(p1->a)
// {
//     do something
//     parameter *p = (parameter *)p1;
//     while (1)
//     {
//         printf("GetCallBack print! \n");
//         sleep(3); // delay 3s
//         p->callback(p->a);
//     }
// }

// extern int SetCallBackFun(int a, pcb callback)
// {
//     printf("SetCallBackFun print! \n");
//     parameter *p = malloc(sizeof(parameter));
//     p->a = 10;
//     p->callback = callback; //create parameter struct

//     pthread_t Thr1; //thread
//     pthread_create(&Thr1, NULL, callback_thread, (void *)p); // Thr1 will do callback_thread((void *)p)
//     pthread_join(Thr1, NULL); // wait for Thr1 to finish
// }

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
    memset(buffer,0,8192*sizeof(char));
    error = recv(ClientFD, buffer, 128 * sizeof(char), 0);
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
        char *FileString = malloc(NewPtr - &buffer[4] + sizeof(char));
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
            memset(buffer,0,8192*sizeof(char));
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
            memset(buffer,0,8192*sizeof(char));
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
            memset(buffer,0,8192*sizeof(char));
            read(JS, &buffer[50], (8192 - 50) * sizeof(char));
            strncpy(buffer, "HTTP/1.1 200 OK\r\nContent-Type: text/javascript\r\n\r\n", 50 * sizeof(char));

            buffer[8191] = '\0'; // to ensure strlen don't get off bounds
            int Diff = strlen(buffer);

            send(ClientFD, buffer, Diff * sizeof(char), 0);
            close(ClientFD);
            return 0;
        }
        else
        {
            // sending error
            send(ClientFD, "HTTP/1.1 204 No Content\r\n", 26 * sizeof(char), 0);
            close(ClientFD);
            return 0;
        }
    }
    else
    {
        // client want method other than GET, server doesn't
        send(ClientFD, "HTTP/1.1 405 Method Not Allowed\r\nContent-Type: text plain\r\n\r\n405 Method Not Allowed", 84 * sizeof(char), 0);
        close(ClientFD);
        return 0;
    }
}