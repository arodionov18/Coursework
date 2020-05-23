#include <iostream>
#include <iomanip>

#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/printer.h>

#include "net.h"

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    LOG(INFO) << "LOG" << std::endl;
    std::cout << "Hello World!" << std::endl;
    std::string network_name = argv[1];
    std::string weights_name = argv[2];

    parser::net::Net network(network_name, true);
    std::string image_path = argv[3];
    network.ReadImage(image_path);
    network.LoadWeights(weights_name, true);
    network.Init();
    auto results = network.GetResults();
    std::cout << "MaxProb: " << std::fixed << std::setprecision(4) << results.first << ", class: " << results.second;
    google::protobuf::ShutdownProtobufLibrary();
}