#include "VisionModel.h"

VisionModel::VisionModel(int serverPort) : _serverPort(serverPort) {
    _pythonServer = new SocketServer();
};

VisionModel::~VisionModel() {
    delete _pythonServer;
};

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

        // Parse received data if we got something
        std::string data(buffer);
        if (!data.empty()) {
            std::cout << "Parsing data: " << data << std::endl;
            
            // Split the string by the '|' delimiter
            std::vector<std::string> tokens;
            size_t pos = 0;
            std::string token;
            
            while ((pos = data.find('|')) != std::string::npos) {
                token = data.substr(0, pos);
                tokens.push_back(token);
                data.erase(0, pos + 1);
            }
            
            // Process tokens in groups of 5 (x1, y1, x2, y2, confidence)
            for (size_t i = 0; i + 4 < tokens.size(); i += 5) {
                try {
                    BoundingBox box;
                    box.x1 = std::stoi(tokens[i]);
                    box.y1 = std::stoi(tokens[i+1]);
                    box.x2 = std::stoi(tokens[i+2]);
                    box.y2 = std::stoi(tokens[i+3]);
                    box.confidence = std::stof(tokens[i+4]);
                    
                    boundingBoxes.push_back(box);
                    std::cout << "Added box: x1=" << box.x1 << ", y1=" << box.y1 
                              << ", x2=" << box.x2 << ", y2=" << box.y2 
                              << ", conf=" << box.confidence << std::endl;
                }
                catch (const std::exception& e) {
                    std::cerr << "Error parsing bounding box at index " << i << ": " << e.what() << std::endl;
                }
            }
        }

        _pythonServer->disconnect();
    }

    return boundingBoxes;
}

void VisionModel::startPythonScript() {
    // Implementation if needed
}
