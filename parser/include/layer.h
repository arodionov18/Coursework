#pragma once

#include "Halide.h"
#include "common.h"
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

using namespace Halide;

class AbstractLayer {
public:
    virtual ~AbstractLayer() = default;
    AbstractLayer(std::weak_ptr<AbstractLayer> input);

    std::weak_ptr<AbstractLayer> input_layer;

    Func forward;
    std::vector<Buffer<float>> params;
    std::vector<Buffer<float>> grads;
    std::vector<Buffer<float>> cache;
    int x, y, z, n;

    std::vector<Func> f_param_grads;
    Func f_in_grad;

    virtual void back_propagate(Func dforward) = 0;

    // Number of output dimensions
    virtual int out_dims() const = 0;

    // Size of output dimension i, 0 <= i < out_dims()
    virtual int out_dim_size(int i) const = 0;
};
/*
class DataLayer : public AbstractLayer {
public:
    DataLayer(const LayerParameter& params);

private:
};

class BatchNormLayer : public AbstractLayer {
public:
    BatchNormLayer(const LayerParameter& params);

private:
};

class ScaleLayer : public AbstractLayer {
public:
    ScaleLayer(const LayerParameter& params);

private:
};

class ReLuLayer : public AbstractLayer {
public:
    ReLuLayer(const LayerParameter& params);

private:
};

class PoolingLayer: public AbstractLayer {
public:
    PoolingLayer(const LayerParameter& params);

private:
};

class InnerProductLayer : public AbstractLayer {
public:
    InnerProductLayer(const LayerParameter& params);

private:
};

class AccuracyLayer : public AbstractLayer {
public:
    AccuracyLayer(const LayerParameter& params);

private:
};


class DropOutLayer : public AbstractLayer {
public:
    DropOutLayer(const LayerParameter& params);

private:
};

class SoftMaxLayer : public AbstractLayer {
public:
    SoftMaxLayer(const LayerParameter& params);

private:
};
*/