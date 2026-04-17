#pragma once
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <memory>

class ModelInference {
public:
    ModelInference(const std::string& model_path);

    std::vector<float> infer(const std::vector<float>& input_tensor);
    std::string predict(const std::vector<float>& input_tensor, float threshold = 0.5f);

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
};