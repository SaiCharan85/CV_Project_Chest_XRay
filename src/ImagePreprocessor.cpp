#include "ImagePreprocessor.h"

std::vector<float> ImagePreprocessor::preprocess(const cv::Mat& image) {
    cv::Mat resized, float_img;

    // resize to 224x224
    cv::resize(image, resized, cv::Size(IMG_SIZE, IMG_SIZE));

    // convert BGR -> RGB
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // convert to float [0, 1]
    resized.convertTo(float_img, CV_32F, 1.0 / 255.0);

    // ImageNet mean and std
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3]  = {0.229f, 0.224f, 0.225f};

    // split into channels and normalize
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    for (int c = 0; c < 3; c++) {
        channels[c] = (channels[c] - mean[c]) / std[c];
    }

    // flatten to CHW vector [3 x 224 x 224]
    std::vector<float> output;
    output.reserve(3 * IMG_SIZE * IMG_SIZE);

    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < IMG_SIZE; h++) {
            for (int w = 0; w < IMG_SIZE; w++) {
                output.push_back(channels[c].at<float>(h, w));
            }
        }
    }

    return output;
}