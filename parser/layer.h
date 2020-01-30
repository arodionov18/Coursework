#pragma once

#include "Halide.h"
#include "generated/caffe.pb.h"
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

using Network::LayerParameter;

class AbstractLayer {
public:
    virtual ~AbstractLayer() = default;
    AbstractLayer();

    Halide::Func forward;
    int x, y, z, w;

protected:
    LayerParameter m_params;
    Halide::Var i, j, k, l;
};

class DataLayer : public AbstractLayer {
public:
    DataLayer(const LayerParameter& params);

private:
    LayerParameter m_params;
};

class ConvolutionalLayer : public AbstractLayer {
public:
    ConvolutionalLayer(const LayerParameter& params);

private:
    LayerParameter m_params;
};

class BatchNormLayer : public AbstractLayer {
public:
    BatchNormLayer(const LayerParameter& params);

private:
    LayerParameter m_params;
};

class ScaleLayer : public AbstractLayer {
public:
    ScaleLayer(const LayerParameter& params);

private:
    LayerParameter m_params;
};

class ReLuLayer : public AbstractLayer {
public:
    ReLuLayer(const LayerParameter& params);

private:
    LayerParameter m_params;
};

class PoolingLayer: public AbstractLayer {
public:
    PoolingLayer(const LayerParameter& params);

private:
    LayerParameter m_params;
};

class InnerProductLayer : public AbstractLayer {
public:
    InnerProductLayer(const LayerParameter& params);

private:
    LayerParameter m_params;
};

class AccuracyLayer : public AbstractLayer {
public:
    AccuracyLayer(const LayerParameter& params);

private:
    LayerParameter m_params;
};


class DropOutLayer : public AbstractLayer {
public:
    DropOutLayer(const LayerParameter& params);

private:
    LayerParameter m_params;
};

class SoftMaxLayer : public AbstractLayer {
public:
    SoftMaxLayer(const LayerParameter& params);

private:
    LayerParameter m_params;
};