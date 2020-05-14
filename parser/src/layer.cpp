#include "layer.h"


//  AbstractLayer
AbstractLayer::AbstractLayer(std::weak_ptr<AbstractLayer> input) {
    if (!input.expired()) {
        //assert(input->forward.defined());

        input_layer = input;
    }
}