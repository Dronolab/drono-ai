#include <iostream>
#include <sstream>
#include <sys/socket.h>    // For socket functions
#include <netinet/in.h>     // For sockaddr_in
#include <arpa/inet.h>      // For inet_pton() if needed
#include <unistd.h>         // For close()
#include <cstring>
#include "VisionModel.h"
#include <unistd.h>

#define PORT 12345 // Port to listen on

int main() {
    VisionModel model(PORT);
    while(1){
        std::cout << "Asking bounding boxes\n";
        std::vector<VisionModel::BoundingBox> boxes = model.getBoundingBoxes();
        sleep(1);
        std::cout << "Received bounding boxes:\n";
        for (const auto& box : boxes) {
            std::cout << "x1: " << box.x1 << ", y1: " << box.y1
                    << ", x2: " << box.x2 << ", y2: " << box.y2
                    << ", confidence: " << box.confidence << "\n";
        }
    }
    return 0;
}
