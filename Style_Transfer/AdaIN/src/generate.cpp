#include <iostream>                    // std::cout, std::flush
#include <fstream>                     // std::ifstream, std::ofstream
#include <filesystem>                  // std::filesystem
#include <string>                      // std::string
#include <sstream>                     // std::stringstream
#include <tuple>                       // std::tuple
#include <vector>                      // std::vector
#include <utility>                     // std::pair
#include <cstdlib>                     // std::exit
// For External Library
#include <torch/torch.h>               // torch
#include <boost/program_options.hpp>   // boost::program_options
// For Original Header
#include "loss.hpp"                    // Loss
#include "networks.hpp"                // MC_VGGNet
#include "visualizer.hpp"              // visualizer
#include "progress.hpp"                // progress

// Define Namespace
namespace fs = std::filesystem;
namespace po = boost::program_options;

// Function Prototype
torch::Tensor load_image(const std::string &path, const int width, const int height, torch::Device &device);
torch::Tensor preprocess(torch::Tensor image);
torch::Tensor postprocess(torch::Tensor image);
std::pair<torch::Tensor, torch::Tensor> calc_mean_std(torch::Tensor feature);
torch::Tensor adaptive_instance_normalization(torch::Tensor content_feature, torch::Tensor style_feature);


// -------------------
// Training Function
// -------------------
void generate(po::variables_map &vm, torch::Device &device){

    constexpr std::string_view extension = "png";  // the extension of file name to save sample images
    constexpr std::pair<float, float> output_range = {0.0, 1.0};  // range of the value in output images

    // -----------------------------------
    // a0. Initialization and Declaration
    // -----------------------------------

    size_t iterations;
    std::string checkpoint_dir, path, result_dir;
    std::vector<std::string> style_layers;
    std::ofstream ofs;
    std::stringstream ss;
    torch::Tensor content_image, style_image, generated;
    torch::Tensor loss, content_loss, style_loss, target_feature, gen_mean, gen_std, style_mean, style_std;
    progress::display *show_progress;
    std::map<std::string, torch::Tensor> content_features, style_features, generated_features;
    MC_VGGNet vgg;


    // -----------------------------------
    // a1. Preparation
    // -----------------------------------

    // (1) Load VGG model
    vgg->to(device);
    torch::load(vgg, vm["vgg_path"].as<std::string>(), device);
    vgg->eval();

    // (2) Extract Features for Content and Style
    {

        torch::NoGradGuard no_grad;

        content_image = load_image("datasets/" + vm["dataset"].as<std::string>() + "/" + vm["content"].as<std::string>(), 0, 0, device);
        style_image = load_image("datasets/" + vm["dataset"].as<std::string>() + "/" + vm["style"].as<std::string>(), content_image.size(3), content_image.size(2), device);
        content_features = vgg->forward(preprocess(content_image.clone()));
        style_features = vgg->forward(preprocess(style_image.clone()));

        target_feature = adaptive_instance_normalization(content_features["relu4_1"], style_features["relu4_1"]);
        
    }

    // (3) Prepare Generated Image
    generated = content_image.detach().clone();
    generated.requires_grad_(true);

    // (4) Set Optimizer Method
    auto optimizer = torch::optim::Adam({generated}, torch::optim::AdamOptions(vm["lr"].as<float>()).betas({vm["beta1"].as<float>(), vm["beta2"].as<float>()}));

    // (5) Set Loss Function
    auto criterion = Loss(vm["loss"].as<std::string>());

    // (6) Make Directories and Open Files
    checkpoint_dir = "checkpoints/" + vm["dataset"].as<std::string>();
    path = checkpoint_dir + "/log";  fs::create_directories(path);
    result_dir = vm["result_dir"].as<std::string>();  fs::create_directories(result_dir);
    ofs.open(checkpoint_dir + "/log/generate.txt", std::ios::out);


    // -----------------------------------
    // a2. Training Image
    // -----------------------------------
    
    // (1) Training
    iterations = vm["iterations"].as<size_t>();
    style_layers = {"relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"};
    show_progress = new progress::display(/*count_max_=*/iterations, /*epoch=*/{1, 1}, /*loss_=*/{"loss"});
    for (size_t iteration = 0; iteration < iterations; iteration++){

        generated_features = vgg->forward(preprocess(generated));

        content_loss = criterion(generated_features["relu4_1"], target_feature.detach());
        style_loss = torch::zeros({}).to(device);
        for (const auto &layer : style_layers){
            std::tie(gen_mean, gen_std) = calc_mean_std(generated_features[layer]);
            std::tie(style_mean, style_std) = calc_mean_std(style_features[layer]);
            style_loss += criterion(gen_mean, style_mean);
            style_loss += criterion(gen_std, style_std);
        }
        loss = vm["content_weight"].as<float>() * content_loss + vm["style_weight"].as<float>() * style_loss;

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        show_progress->increment(/*loss_value=*/{loss.item<float>()});
        ofs << "iters:" << show_progress->get_iters() << '/' << iterations << ' ' << std::flush;
        ofs << "loss:" << loss.item<float>() << std::endl;
        
    }
    delete show_progress;

    // (2) Save Image
    ss.str(""); ss.clear(std::stringstream::goodbit);
    ss << result_dir << "/Generated_Image."  << extension;
    visualizer::save_image(generated.detach(), ss.str(), /*range=*/output_range, /*cols=*/1, /*padding=*/0);

    // Post Processing
    ofs.close();

    // End Processing
    return;

}


// -----------------------
// Image Loader Function
// -----------------------
torch::Tensor load_image(const std::string &path, const int width, const int height, torch::Device &device){

    cv::Mat image, resized;
    torch::Tensor tensor;

    image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error : Couldn't open the image '" << path << "'." << std::endl;
        std::exit(1);
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    if (width == 0){
        resized = image;
    }
    else{
        cv::resize(image, resized, cv::Size(width, height));
    }
    
    tensor = torch::from_blob(resized.data, {resized.rows, resized.cols, 3}, torch::kUInt8).detach().clone();
    tensor = tensor.to(torch::kFloat32) / 255.0;
    tensor = tensor.permute({2, 0, 1}).unsqueeze(0).contiguous();
    tensor = tensor.to(device);

    return tensor;

}


// ---------------------
// Preprocess Function
// ---------------------
torch::Tensor preprocess(torch::Tensor image){
    torch::Tensor mean = torch::tensor({0.485, 0.456, 0.406}).view({1, 3, 1, 1}).to(image.device());
    torch::Tensor std = torch::tensor({0.229, 0.224, 0.225}).view({1, 3, 1, 1}).to(image.device());
    return (image - mean) / std;
}


// ----------------------
// Postprocess Function
// ----------------------
torch::Tensor postprocess(torch::Tensor image){
    torch::Tensor mean = torch::tensor({0.485, 0.456, 0.406}).view({1, 3, 1, 1}).to(image.device());
    torch::Tensor std = torch::tensor({0.229, 0.224, 0.225}).view({1, 3, 1, 1}).to(image.device());
    torch::Tensor out = image * std + mean;
    return out.clamp(0.0, 1.0);
}


// ---------------------------------
// Mean and Standard Deviation Calc
// ---------------------------------
std::pair<torch::Tensor, torch::Tensor> calc_mean_std(torch::Tensor feature){

    torch::Tensor mean, std;

    mean = feature.mean(/*dim=*/{2, 3}, /*keepdim=*/true);
    std = torch::sqrt(feature.var(/*dim=*/{2, 3}, /*unbiased=*/false, /*keepdim=*/true) + 1e-5);

    return {mean, std};

}


// ---------------------------------
// Adaptive Instance Normalization
// ---------------------------------
torch::Tensor adaptive_instance_normalization(torch::Tensor content_feature, torch::Tensor style_feature){

    torch::Tensor content_mean, content_std, style_mean, style_std;

    std::tie(content_mean, content_std) = calc_mean_std(content_feature);
    std::tie(style_mean, style_std) = calc_mean_std(style_feature);

    return style_std * (content_feature - content_mean) / content_std + style_mean;

}