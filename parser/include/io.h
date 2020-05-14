#pragma once

#include "caffe2.pb.h"
#include <glog/logging.h>
#include <google/protobuf/message.h>

namespace parser {
namespace io {

using google::protobuf::Message;
using std::string;

static const int kProtoReadBytesLimit = INT_MAX;

bool ReadProtoFromTextFile(const string& filename, Message* proto);
bool WriteProtoToTextFile(const Message& proto, const string& filename);

bool ReadProtoFromBinaryFile(const string& filename, Message* proto);
void WriteProtoToBinaryFile(const Message& proto, const string& filename);

//void ReadNetParamsFromFile(const string& param_file, bool isBinary, NetParameter* param);
//void ReadNetParamsFromBinaryFile(const string& param_file, NetParameter* param);
// void ReadSolverParamsFromTextFile(const string& param_file, SolverParameter* param);

} // io
} // parser