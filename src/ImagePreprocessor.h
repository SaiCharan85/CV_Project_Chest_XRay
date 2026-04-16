#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

class ImagePreprocessor {
public:
    // Resize to 224x224, normalize with ImageNet mean/std
    // Returns flat float vector: [3 x 224 x 224] in CHW format
    static std::vector<float> preprocess(const cv::Mat& image);

    static const int IMG_SIZE = 224;
};