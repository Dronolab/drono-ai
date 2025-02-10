#include "SocketServer.h"

SocketServer::SocketServer(){
    
}

SocketServer::~SocketServer() {

}

bool SocketServer::connect(const int port, const char* ip) { 

    // Create socket
    _sock = socket(AF_INET, SOCK_STREAM, 0);
    if (_sock < 0) {
        std::cerr << "Error: Could not create socket\n";
        return -1;
    }

    
    _serverAddr.sin_family = AF_INET;
    _serverAddr.sin_port = htons(port);
    inet_pton(AF_INET, ip, &_serverAddr.sin_addr);

    // Connect to server
    if (::connect(_sock, (struct sockaddr*)&_serverAddr, sizeof(_serverAddr)) < 0) {
        std::cerr << "Error: Could not connect to server\n";
        close(_sock);
        return -1;
    }
    return 1;
}

bool SocketServer::disconnect(){
    close(_sock);
    return 1;   
}

void SocketServer::send(const char* request, int len) {
    ::send(_sock, request, strlen(request), 0);
}

void SocketServer::read(char * buffer, int len) {
    recv(_sock, buffer, sizeof(buffer), 0);
}

