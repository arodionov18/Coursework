#include <iostream>

#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/printer.h>

#include "net.h"

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    std::cout << "Hello World!" << std::endl;
    std::string network_name = argv[1];
    std::string weights_name = argv[2];

    parser::net::Net network(network_name, true);
    std::string image_path = argv[3];
    network.ReadImage(image_path);
    network.LoadWeights(weights_name, true);
    network.Init();

    google::protobuf::ShutdownProtobufLibrary();
}