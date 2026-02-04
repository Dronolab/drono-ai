#include <cstdio>
#include <iostream>
#include <string>
#include <thread>

int main() {
    FILE* pipe = 
    #ifdef _WIN32
        _popen("python test.py", "r");
    #else
        popen("python3 test.py", "r");
    #endif

    if(!pipe) {
        std::cerr << "Failed to open python script." << std::endl;
        return 1;
    }

    char buffer[256];
    std::atomic<bool> stop(false);

    std::thread inputThread([&stop]() {
        char c;
        while (std::cin >> c) {
            if (c == 'q') {
                stop = true;
                break;
            }
        }
    });

    std::cout << "Press 'q' + Enter to quit.\n";

    while (!stop && fgets(buffer, sizeof(buffer), pipe)) {
        std::cout << buffer;       // print immediately
        std::cout.flush();         // flush C++ buffer
    }

    #ifdef _WIN32
        _pclose(pipe);
    #else
        pclose(pipe);
    #endif

    inputThread.join();

    std::cout << "\nC++ listener stopped.\n";
    return 0;
}