#include "VisionModel.h"

VisionModel::VisionModel(int serverPort) : _serverPort(serverPort) {
    _pythonServer = new SocketServer();
};

VisionModel::~VisionModel() {};

std::vector<VisionModel::BoundingBox> VisionModel::getBoundingBoxes() {
    std::vector<VisionModel::BoundingBox> boundingBoxes;
    
    bool connected = _pythonServer->connect(_serverPort, "127.0.0.1");

    if(connected) {
        // Send request for bounding boxes
        const char* request = "Request bounding boxes";
        _pythonServer->send(request, strlen(request));

        // Receive bounding boxes data
        char buffer[1024] = {0};
        _pythonServer->read(buffer, sizeof(buffer));

        _pythonServer->disconnect();

        // Parse received data
        std::stringstream ss(buffer);
        BoundingBox bb;
        char delimiter;

        while (ss >> bb.x1 >> delimiter >> bb.y1 >> delimiter >> bb.x2 >> delimiter >> bb.y2 >> delimiter >> bb.confidence >> delimiter) {
            boundingBoxes.push_back(bb);
        }
    }

    

    return boundingBoxes;
}

void VisionModel::startPythonScript(){

}
