#include <sys/socket.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "./handle.h"

#define PORT 2500
#define BLOG 6

int main(void)
{
    int SocketFD;
    SocketFD = socket(AF_INET, SOCK_STREAM, 0);
    if (SocketFD == -1)
    {
        perror("socket");
        return -1;
    }

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

    // char buffer[512];

    // int i = 3;
    //     ClientFD = accept(SocketFD,NULL,NULL);
    //     if(ClientFD < 0){
    //         perror("accept");
    //         i--;
    //     }
    //     if(recv(ClientFD,buffer,128*sizeof(char),0)==-1){
    //         perror("recv");
    //         i--;
    //     }
    //     printf("message came from %d which was:\n%s\n",ClientFD,buffer);
    Handle(SocketFD);
    Handle(SocketFD);
    Handle(SocketFD);
    sleep(1);
    close(SocketFD);
    printf("server has closed successfully\n");
    return 0;
}