#pragma once
#include <opencv2/opencv.hpp>
#include <string>

class Visualizer {
public:
    // constructor takes the output folder path
    Visualizer(const std::string& output_dir);

    // draws prediction + confidence on the image and saves it
    void annotateAndSave(const cv::Mat& image,
                         const std::string& filename,
                         const std::string& prediction,
                         float confidence);

private:
    std::string output_dir_;
};