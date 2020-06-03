#include <iostream>
#include <iomanip>

#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/printer.h>
#include <chrono>

#include "net.h"

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    LOG(INFO) << "LOG" << std::endl;
    std::cerr << "Hello World!" << std::endl;
    std::string network_name = argv[1];
    std::string weights_name = argv[2];

    auto start_time = std::chrono::high_resolution_clock::now();
    parser::net::Net network(network_name, true);
    std::string image_path = argv[3];
    network.ReadImage(image_path);
    auto start_load_weights = std::chrono::high_resolution_clock::now();
    network.LoadWeights(weights_name, true);
    auto finished_load_weights = std::chrono::high_resolution_clock::now();
    network.Init();
    auto construction_finished_time = std::chrono::high_resolution_clock::now();
    std::cout << "Network is ready, time for construction: " <<  std::endl;
    std::cout << "\t-Loading weights: " << std::chrono::duration_cast<std::chrono::milliseconds>(finished_load_weights - start_load_weights).count() << " milliseconds" << std::endl;
    std::cout << "\t-Overall: " << std::chrono::duration_cast<std::chrono::milliseconds>(finished_load_weights - start_time).count() << " milliseconds" << std::endl;
    auto start_proccessing = std::chrono::high_resolution_clock::now();
    auto results = network.GetResults();
    auto finished_proccessing = std::chrono::high_resolution_clock::now();
    std::cout << "Time for proccessing image: " << std::chrono::duration_cast<std::chrono::milliseconds>( finished_proccessing - start_proccessing).count() << " milliseconds" << std::endl;
    std::cout << "MaxProb: " << std::fixed << std::setprecision(4) << results.first << ", class: " << results.second << std::endl;
    google::protobuf::ShutdownProtobufLibrary();
}