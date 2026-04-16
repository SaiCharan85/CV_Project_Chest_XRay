#include <iostream>
#include "ImageLoader.h"
#include "ImagePreprocessor.h"

int main(int argc, char* argv[]) {
    std::string dataset_root = "chest_xray/chest_xray";

    // Load test set
    auto images = ImageLoader::loadDataset(dataset_root, "test");

    if (images.empty()) {
        std::cerr << "No images loaded. Check dataset path." << std::endl;
        return 1;
    }

    // Test preprocessing on first image
    auto& first = images[0];
    auto tensor = ImagePreprocessor::preprocess(first.image);

    std::cout << "Preprocessed: " << first.filename
              << " | label: " << first.label
              << " | tensor size: " << tensor.size()
              << " | first values: "
              << tensor[0] << ", " << tensor[1] << ", " << tensor[2]
              << std::endl;

    // ModelInference will go here (Person 2)

    return 0;
}