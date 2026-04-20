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
    const std::string& actual_label,
    const std::string& prediction,
    float confidence) {

    cv::Mat annotated = image.clone();

    bool correct = (actual_label == prediction);

    // --- Top banner: Actual (ground truth) label ---
    // always use a dark blue/gray banner for the actual label
    cv::Scalar actual_color(140, 80, 30);  // dark blue-gray (BGR)
    cv::rectangle(annotated, cv::Point(0, 0),
        cv::Point(annotated.cols, 45),
        actual_color, cv::FILLED);

    std::string actual_text = "Actual: " + actual_label;
    cv::putText(annotated, actual_text,
        cv::Point(10, 30),
        cv::FONT_HERSHEY_SIMPLEX, 0.8,
        cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    // --- Bottom banner: Predicted label + confidence ---
    // green if correct, red if wrong
    cv::Scalar pred_color;
    if (correct) {
        pred_color = cv::Scalar(0, 180, 0);    // green (BGR)
    }
    else {
        pred_color = cv::Scalar(0, 0, 220);    // red (BGR)
    }

    int banner_h = 55;
    int banner_top = annotated.rows - banner_h;
    cv::rectangle(annotated, cv::Point(0, banner_top),
        cv::Point(annotated.cols, annotated.rows),
        pred_color, cv::FILLED);

    // prediction text on the left
    std::string pred_text = "Predicted: " + prediction;
    cv::putText(annotated, pred_text,
        cv::Point(10, banner_top + 22),
        cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    // confidence text below prediction
    std::ostringstream conf_text;
    conf_text << "Confidence: " << std::fixed << std::setprecision(1)
        << (confidence * 100.0f) << "%";
    cv::putText(annotated, conf_text.str(),
        cv::Point(10, banner_top + 45),
        cv::FONT_HERSHEY_SIMPLEX, 0.55,
        cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

    // match/mismatch icon on the right side of bottom banner
    std::string status = correct ? "CORRECT" : "WRONG";
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(status, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
    cv::putText(annotated, status,
        cv::Point(annotated.cols - text_size.width - 15, banner_top + 33),
        cv::FONT_HERSHEY_SIMPLEX, 0.7,
        cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    // save to output folder
    std::string save_path = output_dir_ + "/" + filename;
    if (!cv::imwrite(save_path, annotated)) {
        std::cerr << "Failed to save: " << save_path << std::endl;
    }
    else {
        std::cout << "Saved: " << save_path << std::endl;
    }
}