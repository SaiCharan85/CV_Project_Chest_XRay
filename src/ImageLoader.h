#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct ImageData {
    cv::Mat image;
    std::string filename;
    int label; // 0 = NORMAL, 1 = PNEUMONIA
};

class ImageLoader {
public:
    // Load all images from a folder, label = 0 or 1
    static std::vector<ImageData> loadFromFolder(const std::string& folder_path, int label);

    // Load both NORMAL and PNEUMONIA from a split (e.g. "test")
    static std::vector<ImageData> loadDataset(const std::string& dataset_root, const std::string& split);
};