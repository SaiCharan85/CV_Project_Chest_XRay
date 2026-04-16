#include "ImageLoader.h"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

std::vector<ImageData> ImageLoader::loadFromFolder(const std::string& folder_path, int label) {
    std::vector<ImageData> images;

    if (!fs::exists(folder_path)) {
        std::cerr << "Folder not found: " << folder_path << std::endl;
        return images;
    }

    for (const auto& entry : fs::directory_iterator(folder_path)) {
        std::string path = entry.path().string();
        std::string ext  = entry.path().extension().string();

        // skip non-image files
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;

        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Failed to load: " << path << std::endl;
            continue;
        }

        ImageData data;
        data.image    = img;
        data.filename = entry.path().filename().string();
        data.label    = label;
        images.push_back(data);
    }

    std::cout << "Loaded " << images.size() << " images from " << folder_path << std::endl;
    return images;
}

std::vector<ImageData> ImageLoader::loadDataset(const std::string& dataset_root, const std::string& split) {
    std::string normal_path    = dataset_root + "/" + split + "/NORMAL";
    std::string pneumonia_path = dataset_root + "/" + split + "/PNEUMONIA";

    auto normal    = loadFromFolder(normal_path,    0);
    auto pneumonia = loadFromFolder(pneumonia_path, 1);

    // merge
    normal.insert(normal.end(), pneumonia.begin(), pneumonia.end());
    std::cout << "Total " << split << " images: " << normal.size() << std::endl;
    return normal;
}