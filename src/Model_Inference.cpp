#include "Model_Inference.h"
#include <cmath>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

// ---------------------------------------------------------------
// Constructor — loads the ONNX model into an ORT session
// ---------------------------------------------------------------
ModelInference::ModelInference(const std::string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "PneumoniaDetector")
{
    // Check if file exists first
    if (!fs::exists(model_path)) {
        throw std::runtime_error("Model file not found: " + model_path);
    }

    // Configure session options BEFORE creating session
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    // Now create the session
    std::wstring wide_path(model_path.begin(), model_path.end());
    session = std::make_unique<Ort::Session>(env, wide_path.c_str(), session_options);

    std::cout << "Model loaded: " << model_path << std::endl;
}

// ---------------------------------------------------------------
// infer() — feeds a flat CHW tensor through the model
// ---------------------------------------------------------------
std::vector<float> ModelInference::infer(const std::vector<float>& input_tensor) {

    std::vector<int64_t> input_shape = { 1, 3, 224, 224 };

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault
    );

    Ort::Value input_ort = Ort::Value::CreateTensor<float>(
        mem_info,
        const_cast<float*>(input_tensor.data()),
        input_tensor.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[] = { "input" };
    const char* output_names[] = { "output" };

    auto output_tensors = session->Run(
        Ort::RunOptions{ nullptr },
        input_names,
        &input_ort,
        1,
        output_names,
        1
    );

    float* output_data = output_tensors[0].GetTensorMutableData<float>();

    return { output_data[0], output_data[1] };
}

// ---------------------------------------------------------------
// predict() — runs infer(), applies softmax, thresholds
// ---------------------------------------------------------------
std::string ModelInference::predict(const std::vector<float>& input_tensor, float threshold) {

    std::vector<float> logits = infer(input_tensor);

    float max_logit = std::max(logits[0], logits[1]);
    float exp_normal = std::exp(logits[0] - max_logit);
    float exp_pneumonia = std::exp(logits[1] - max_logit);
    float sum_exp = exp_normal + exp_pneumonia;

    float prob_pneumonia = exp_pneumonia / sum_exp;

    if (prob_pneumonia >= threshold) {
        return "PNEUMONIA";
    }
    else {
        return "NORMAL";
    }
}