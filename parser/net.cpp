#include "net.h"
#include "io.h"
#include "layer.h"
#include <iostream>

namespace parser {
namespace net {

using Network::LayerParameter;

unique_ptr<AbstractLayer> ParseAndCreateLayer(const LayerParameter& parameters) {
    CHECK(!parameters.has_type()) << "Wrong layer";
    if (parameters.type() == "DATA")
            return std::make_unique<DataLayer>(parameters);
    if (parameters.type() == "CONVOLUTIONAL")
        return std::make_unique<ConvolutionalLayer>(parameters);
    if (parameters.type() == "RELU")
        return std::make_unique<ReLuLayer>(parameters);
    if (parameters.type() == "POOLING")
        return std::make_unique<PoolingLayer>(parameters);
    if (parameters.type() == "INNER_PRODUCT")    
        return std::make_unique<InnerProductLayer>(parameters);
    if (parameters.type() == "ACCURACY")    
        return std::make_unique<AccuracyLayer>(parameters);
    if (parameters.type() == "DROPOUT")
        return std::make_unique<DropOutLayer>(parameters);
    if (parameters.type() == "SOFTMAX_LOSS")    
        return std::make_unique<SoftMaxLayer>(parameters);

    std::cerr << "ERROR: UNKNOWN LAYER TYPE " << parameters.type() << " WITH NAME " << parameters.name() << std::endl;
    return std::make_unique<AbstractLayer>();
}

Net::Net(const NetParameter& parameters) {
    init(parameters);
}

Net::Net(const string& filename, bool binary) {
    NetParameter parameters;
    // TODO: ReadNetParametersFromFile(filename, binary);
    init(parameters);
}

void Net::init(const NetParameter& parameters) {
    this->name = parameters.name();
    // TODO
    for (size_t i = 0; i < parameters.layer_size(); ++i) {
        auto& layer = parameters.layer(i);
        this->layers.push_back(ParseAndCreateLayer(layer));
    }
}

} // net
} // parser