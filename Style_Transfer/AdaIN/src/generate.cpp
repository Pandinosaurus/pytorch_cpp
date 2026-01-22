#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <sstream>
#include <tuple>
#include <vector>
#include <utility>
#include <cstdlib>
// For External Library
#include <torch/torch.h>
#include <boost/program_options.hpp>
// For Original Header
#include "loss.hpp"
#include "networks.hpp"
#include "visualizer.hpp"
#include "progress.hpp"

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

    constexpr std::string_view extension = "png";
    constexpr std::pair<float, float> output_range = {0.0, 1.0};

    // -----------------------------------
    // a0. Initialization and Declaration
    // -----------------------------------

    size_t iterations;
    std::string checkpoint_dir, path, result_dir;
    std::vector<std::string> style_layers;
    std::ofstream ofs;
    std::stringstream ss;
    torch::Tensor content_image, style_image, generated;
    torch::Tensor loss, content_loss, style_loss, target_feature, adain_output;
    torch::Tensor gen_mean, gen_std, style_mean, style_std;
    progress::display *show_progress;
    std::map<std::string, torch::Tensor> content_features, style_features, generated_features;
    MC_VGGNet encoder;
    Decoder decoder;


    // -----------------------------------
    // a1. Preparation
    // -----------------------------------

    // (1) Load VGG Encoder
    encoder->to(device);
    torch::load(encoder, vm["vgg_path"].as<std::string>(), device);
    encoder->eval();  // Encoder is fixed (not trained)
    
    // Freeze encoder parameters
    for (auto& param : encoder->parameters()) {
        param.requires_grad_(false);
    }

    // (2) Initialize Decoder
    decoder->to(device);
    decoder->train();  // Decoder will be trained

    // (3) Extract Features for Content and Style (Fixed)
    torch::Tensor content_feature_relu4_1, style_feature_relu4_1;
    {
        torch::NoGradGuard no_grad;

        content_image = load_image("datasets/" + vm["dataset"].as<std::string>() + "/" + vm["content"].as<std::string>(), 0, 0, device);
        style_image = load_image("datasets/" + vm["dataset"].as<std::string>() + "/" + vm["style"].as<std::string>(), content_image.size(3), content_image.size(2), device);
        
        content_features = encoder->forward(preprocess(content_image.clone()));
        style_features = encoder->forward(preprocess(style_image.clone()));
        
        content_feature_relu4_1 = content_features["relu4_1"];
        style_feature_relu4_1 = style_features["relu4_1"];
        
    }

    // (4) Set Optimizer Method (for Decoder only)
    auto optimizer = torch::optim::Adam(decoder->parameters(), torch::optim::AdamOptions(vm["lr"].as<float>()).betas({vm["beta1"].as<float>(), vm["beta2"].as<float>()}));

    // (5) Set Loss Function
    auto criterion = Loss(vm["loss"].as<std::string>());

    // (6) Make Directories and Open Files
    checkpoint_dir = "checkpoints/" + vm["dataset"].as<std::string>();
    path = checkpoint_dir + "/log";  fs::create_directories(path);
    result_dir = vm["result_dir"].as<std::string>();  fs::create_directories(result_dir);
    ofs.open(checkpoint_dir + "/log/generate.txt", std::ios::out);


    // -----------------------------------
    // a2. Training Decoder
    // -----------------------------------
    
    // (1) Training Loop
    iterations = vm["iterations"].as<size_t>();
    style_layers = {"relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"};
    show_progress = new progress::display(/*count_max_=*/iterations, /*epoch=*/{1, 1}, /*loss_=*/{"loss"});
    
    for (size_t iteration = 0; iteration < iterations; iteration++){

        // Apply AdaIN
        target_feature = adaptive_instance_normalization(content_feature_relu4_1, style_feature_relu4_1);
        
        // Decode
        generated = decoder->forward(target_feature);

        // Extract features from generated image
        generated_features = encoder->forward(generated);

        // Content Loss (in feature space)
        content_loss = criterion(generated_features["relu4_1"], target_feature.detach());

        // Style Loss (match mean and std at multiple layers)
        style_loss = torch::zeros({}).to(device);
        for (const auto &layer : style_layers){
            std::tie(gen_mean, gen_std) = calc_mean_std(generated_features[layer]);
            std::tie(style_mean, style_std) = calc_mean_std(style_features[layer]);
            style_loss += criterion(gen_mean, style_mean);
            style_loss += criterion(gen_std, style_std);
        }

        // Total Loss
        loss = vm["content_weight"].as<float>() * content_loss + vm["style_weight"].as<float>() * style_loss;

        // Backward and optimize
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        show_progress->increment(/*loss_value=*/{loss.item<float>()});
        ofs << "iters:" << show_progress->get_iters() << '/' << iterations << ' ' << std::flush;
        ofs << "loss:" << loss.item<float>() << " content:" << content_loss.item<float>() << " style:" << style_loss.item<float>() << std::endl;
        
    }
    delete show_progress;

    // (2) Generate Final Image
    decoder->eval();
    {
        torch::NoGradGuard no_grad;
        target_feature = adaptive_instance_normalization(content_feature_relu4_1, style_feature_relu4_1);
        generated = decoder->forward(target_feature);
        generated = postprocess(generated);
    }

    // (3) Save Image
    ss.str(""); ss.clear(std::stringstream::goodbit);
    ss << result_dir << "/Generated_Image." << extension;
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