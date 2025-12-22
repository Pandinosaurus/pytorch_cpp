#include <iostream>
#include <string>
#include <vector>
#include <typeinfo>
#include <cstdlib>
// For External Library
#include <torch/torch.h>
// For Original Header
#include "networks.hpp"

// Define Namespace
namespace nn = torch::nn;
namespace F = torch::nn::functional;

#define MAX_POOLING -1


// ----------------------------------------------------------------------
// struct{MC_VGGNetImpl}(nn::Module) -> constructor
// ----------------------------------------------------------------------
MC_VGGNetImpl::MC_VGGNetImpl(){

    std::vector<long int> cfg;
    cfg = {64, 64, MAX_POOLING, 128, 128, MAX_POOLING, 256, 256, 256, 256, MAX_POOLING, 512, 512, 512, 512, MAX_POOLING, 512, 512, 512, 512, MAX_POOLING};
    this->layer_names = {
        "conv1_1", "bn1_1", "relu1_1", "conv1_2", "bn1_2", "relu1_2", "pool1",
        "conv2_1", "bn2_1", "relu2_1", "conv2_2", "bn2_2", "relu2_2", "pool2",
        "conv3_1", "bn3_1", "relu3_1", "conv3_2", "bn3_2", "relu3_2", "conv3_3", "bn3_3", "relu3_3", "conv3_4", "bn3_4", "relu3_4", "pool3",
        "conv4_1", "bn4_1", "relu4_1", "conv4_2", "bn4_2", "relu4_2", "conv4_3", "bn4_3", "relu4_3", "conv4_4", "bn4_4", "relu4_4", "pool4",
        "conv5_1", "bn5_1", "relu5_1", "conv5_2", "bn5_2", "relu5_2", "conv5_3", "bn5_3", "relu5_3", "conv5_4", "bn5_4", "relu5_4", "pool5"
    };

    this->features = make_layers(3, cfg, true);  // {C,224,224} ===> {512,7,7}
    register_module("features", this->features);

    this->avgpool = nn::Sequential(nn::AdaptiveAvgPool2d(nn::AdaptiveAvgPool2dOptions({7, 7})));  // {512,X,X} ===> {512,7,7}
    register_module("avgpool", this->avgpool);

    this->classifier = nn::Sequential(
        nn::Linear(/*in_channels=*/512*7*7, /*out_channels=*/4096),                      // {512*7*7} ===> {4096}
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Dropout(0.5),
        nn::Linear(/*in_channels=*/4096, /*out_channels=*/4096),                         // {4096} ===> {4096}
        nn::ReLU(nn::ReLUOptions().inplace(true)),
        nn::Dropout(0.5),
        nn::Linear(/*in_channels=*/4096, /*out_channels=*/1000)  // {4096} ===> {CN}
    );
    register_module("classifier", this->classifier);

}


// ---------------------------------------------------------
// struct{MC_VGGNetImpl}(nn::Module) -> function{forward}
// ---------------------------------------------------------
std::map<std::string, torch::Tensor> MC_VGGNetImpl::forward(torch::Tensor x){

    size_t i;
    std::string name;
    std::map<std::string, torch::Tensor> outputs{};

    i = 0;
    for (auto &m : *features){
        x = m.forward<torch::Tensor>(x);
        name = this->layer_names[i++];
        if ((name == "relu1_1") || (name == "relu2_1") || (name == "relu3_1") || (name == "relu4_1") || (name == "relu5_1")){
            outputs[name] = x;
        }
    }

    return outputs;

}


// ----------------------------
// function{make_layers}
// ----------------------------
nn::Sequential make_layers(const size_t nc, const std::vector<long int> cfg, const bool BN){
    nn::Sequential sq;
    long int in_channels = (long int)nc;
    for (auto v : cfg){
        if (v == MAX_POOLING){
            sq->push_back(nn::MaxPool2d(nn::MaxPool2dOptions(/*kernel_size=*/2).stride(2)));
        }
        else{
            sq->push_back(nn::Conv2d(nn::Conv2dOptions(/*in_channels=*/in_channels, /*out_channels=*/v, /*kernel_size=*/3).stride(1).padding(1)));
            if (BN){
                sq->push_back(nn::BatchNorm2d(v));
            }
            sq->push_back(nn::ReLU(nn::ReLUOptions().inplace(true)));
            in_channels = v;
        }
    }
    return sq;
}
