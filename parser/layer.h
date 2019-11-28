#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

struct AbstractLayerParams {
    std::string name_;
    std::string type_;
    std::string prev_layer_;
    std::string next_layer_;
};


class AbstractLayer {
public:
    virtual ~AbstractLayer() = default;
    AbstractLayer(const AbstractLayerParams& params) : params_(params) {
    }

private:
    AbstractLayerParams params_;
};


struct WeightFiller {
    std::string type;
    double std;
};

struct ConvolutionalParams : public AbstractLayerParams {
    bool bias_term;
    int num_output;
    int pad;
    int kernel_size;
    int stride;
    WeightFiller weight_filler;
};

struct BatchNormParams : public AbstractLayerParams {
    bool use_global_stats;
    double eps;
};

struct ScaleParams : public AbstractLayerParams {
    bool bias_term;
};

enum Pool { MAX };

struct PoolingParams : public AbstractLayerParams {
    Pool pool;
    int kernel_size = 3;
    int stride = 2;
};

class ConvolutionalLayer : public AbstractLayer {
public:

private:
    
    AbstractLayerParams params_;
};