#ifndef NETWORKS_HPP
#define NETWORKS_HPP

#include <string>
#include <vector>
// For External Library
#include <torch/torch.h>
#include <boost/program_options.hpp>

// Define Namespace
namespace nn = torch::nn;
namespace po = boost::program_options;

// Function Prototype
nn::Sequential make_layers(const size_t nc, const std::vector<long int> cfg, const bool BN);


// -------------------------------------------------
// struct{MC_VGGNetImpl}(nn::Module)
// -------------------------------------------------
struct MC_VGGNetImpl : nn::Module{
private:
    std::vector<std::string> layer_names;
    nn::Sequential features, avgpool, classifier;
public:
    MC_VGGNetImpl();
    std::map<std::string, torch::Tensor> forward(torch::Tensor x);
};
TORCH_MODULE(MC_VGGNet);

// -------------------------------------------------
// struct{DecoderImpl}(nn::Module) - Decoder
// -------------------------------------------------
struct DecoderImpl : nn::Module{
private:
    nn::Conv2d conv4_1{nullptr}, conv3_4{nullptr}, conv3_3{nullptr}, conv3_2{nullptr}, conv3_1{nullptr};
    nn::Conv2d conv2_2{nullptr}, conv2_1{nullptr}, conv1_2{nullptr}, conv1_1{nullptr};
    nn::ReLU relu{nullptr};
    nn::ReflectionPad2d pad{nullptr};
    nn::Upsample upsample{nullptr};
public:
    DecoderImpl();
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(Decoder);


#endif