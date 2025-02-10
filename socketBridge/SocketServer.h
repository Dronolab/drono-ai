#pragma once

#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "netinet/in.h"
#include <cstring>

class SocketServer {
    public: 
        SocketServer();
        ~SocketServer();

        bool connect(int port);
        bool disconnect();
        void send(const char * request, int len);
        void read(char * buffer, int len);

    private:
        struct sockaddr_in _serverAddr;
        int _sock;

};