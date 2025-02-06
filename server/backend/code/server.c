#include <sys/socket.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include "../header/handle.h"

#define PORT 2500

uint16_t Working = 0;

int main(void)
{
    pthread_t queuer;
    int queuerId;
    char CommandCharecter = 'I';
    char* fN = "../../frontend/version11.mn1";
    FromFile(fN);
    int SocketFD;
    SocketFD = socket(AF_INET, SOCK_STREAM, 0);
    if (SocketFD == -1)
    {
        perror("socket");
        return -1;
    }
    int opt = 1;
    setsockopt(SocketFD, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    struct sockaddr_in ServerAdress;
    ServerAdress.sin_family = AF_INET;
    ServerAdress.sin_port = htons(PORT);
    inet_pton(AF_INET, "127.0.0.1", &ServerAdress.sin_addr.s_addr);

    if (bind(SocketFD, (struct sockaddr *)&ServerAdress, sizeof(ServerAdress)) == -1)
    {
        perror("bind");
        return -2;
    }

    if (listen(SocketFD, BLOG) == -1)
    {
        perror("listen");
        return -3;
    }
    EnqueuerArguments Args = {
        .ServerFD = SocketFD,
        .ThreadId = 0
    };
    SetBit(Working,0);
    queuerId = pthread_create(&queuer,NULL,Enqueuer,&Args);
    scanf("%c",&CommandCharecter);
    while(CommandCharecter != 'C'){
        scanf("%c",&CommandCharecter);
    }
    pthread_join(queuerId,NULL);
    close(SocketFD);
    printf("server has closed successfully\n");
    return 0;
}