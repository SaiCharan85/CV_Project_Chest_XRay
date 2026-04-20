#include "Visualizer.h"
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

Visualizer::Visualizer(const std::string& output_dir) {
    output_dir_ = output_dir;

    // create output folder if it doesn't exist
    if (!fs::exists(output_dir_)) {
        fs::create_directories(output_dir_);
        std::cout << "Created output folder: " << output_dir_ << std::endl;
    }
}

void Visualizer::annotateAndSave(const cv::Mat& image,
                                  const std::string& filename,
                                  const std::string& prediction,
                                  float confidence) {

    cv::Mat annotated = image.clone();

    // green for normal, red for pneumonia
    cv::Scalar color;
    if (prediction == "NORMAL") {
        color = cv::Scalar(0, 200, 0);
    } else {
        color = cv::Scalar(0, 0, 220);
    }

    // draw a filled rectangle at the top as a label banner
    cv::rectangle(annotated, cv::Point(0, 0),
                  cv::Point(annotated.cols, 50),
                  color, cv::FILLED);

    // write the prediction label on the banner
    cv::putText(annotated, prediction,
                cv::Point(10, 33),
                cv::FONT_HERSHEY_SIMPLEX, 1.0,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    // write confidence score at the bottom
    std::ostringstream conf_text;
    conf_text << "Confidence: " << std::fixed << std::setprecision(1)
              << (confidence * 100.0f) << "%";

    cv::putText(annotated, conf_text.str(),
                cv::Point(10, annotated.rows - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    // save to output folder
    std::string save_path = output_dir_ + "/" + filename;
    if (!cv::imwrite(save_path, annotated)) {
        std::cerr << "Failed to save: " << save_path << std::endl;
    } else {
        std::cout << "Saved: " << save_path << std::endl;
    }
}
