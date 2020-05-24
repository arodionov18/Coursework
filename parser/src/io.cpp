#include "io.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <string>
#include <vector>

namespace parser {
namespace io {

using google::protobuf::io::IstreamInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::OstreamOutputStream;
using google::protobuf::io::CodedOutputStream;

using std::ios;
using std::fstream;
using std::endl;
using std::cerr;

bool ReadProtoFromTextFile(const string& filename, Message* proto) {
    fstream in(filename, ios::in);
    if (!in.is_open()) {
        cerr << "Failed to open " << filename << " for reading" << endl;
    }

    IstreamInputStream inputStream(&in);
    bool result = google::protobuf::TextFormat::Parse(&inputStream, proto);
    in.close();
    return result;
}

bool WriteProtoToTextFile(const string& filename, const Message& proto) {
    fstream out(filename, ios::out | ios::trunc);
    if (!out.is_open()) {
        cerr << "Failed to open " << filename << " for writing" << endl;
    }

    OstreamOutputStream outputStream(&out);
    bool result = google::protobuf::TextFormat::Print(proto, &outputStream);
    out.close();
    return result;
}

bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
    fstream in(filename, ios::in | ios::binary);
    if (!in.is_open()) {
        cerr << "Failed to open " << filename << " for reading" << endl;
    }

    IstreamInputStream inputStream(&in);
    CodedInputStream codedInputStream(&inputStream);
    bool result = proto->ParseFromCodedStream(&codedInputStream);
    in.close();
    return result;
}

bool WriteProtoToBinaryFile(const string& filename, const Message& proto) {
    fstream out(filename, ios::out | ios::trunc | ios::binary);
    if (!out.is_open()) {
        cerr << "Failed to open " << filename << " for writing" << endl;
    }

    bool result = proto.SerializeToOstream(&out);
    out.close();
    return result;
}

void ReadNetParamsFromFile(const string& filename, bool isBinary, NetParameter* parameters) {
    if (isBinary) {
        CHECK(ReadProtoFromBinaryFile(filename, parameters)) << "Failed to parse NetParameter file: " << filename;
    } else {
        CHECK(ReadProtoFromTextFile(filename, parameters)) << "Failed to parse NetParameter file: " << filename;
    }
}


} // io
} // parser