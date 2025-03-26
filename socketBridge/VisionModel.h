#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>

#include "SocketServer.h"


class VisionModel {

    private:
        SocketServer * _pythonServer;
        int _serverPort;
        void startPythonScript();
    public:
        struct BoundingBox {
            int x1, y1, x2, y2;
            float confidence;
        };
        VisionModel(int serverPort);

        ~VisionModel();

        std::vector<BoundingBox> getBoundingBoxes();

};