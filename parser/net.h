#pragma once

#include "generated/caffe.pb.h"
#include "layer.h"

#include <vector>
#include <memory>

namespace parser {
namespace net {

using std::string;
using std::vector;
using std::unique_ptr;
using Network::NetParameter;

class Net {
public:
    explicit Net(const NetParameter& parameters);
    explicit Net(const string& filename, bool binary);

    string name;

protected:
    vector<unique_ptr<AbstractLayer>> layers;

private:
    void init(const NetParameter& parameters);
};

} // net
} // parser