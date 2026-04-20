#include <iostream>
#include <vector>
#include <chrono>
#include "ImageLoader.h"
#include "ImagePreprocessor.h"
#include "Model_Inference.h"
#include "MetricsComputer.h"
#include "Visualizer.h"

int main() {
    std::string dataset_root = "chest_xray/chest_xray";
    std::string model_path = "pneumonia_detector.onnx";
    float threshold = 0.5f;

    // ---- Sushma's part: load the test set ----
    auto images = ImageLoader::loadDataset(dataset_root, "test");

    if (images.empty()) {
        std::cerr << "No images loaded. Check dataset path." << std::endl;
        return 1;
    }

    // ---- Charan's part: load model + run batch inference ----
    try {
        std::cout << "Loading model from: " << model_path << std::endl;
        ModelInference model(model_path);
        std::cout << "Model loaded successfully!" << std::endl;

        // Vectors to store results — Dina will use these for MetricsComputer
        std::vector<int> true_labels;
        std::vector<int> predicted_labels;

        int correct = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < images.size(); i++) {
            auto tensor = ImagePreprocessor::preprocess(images[i].image);
            std::string prediction = model.predict(tensor, threshold);
            int pred_label = (prediction == "PNEUMONIA") ? 1 : 0;

            true_labels.push_back(images[i].label);
            predicted_labels.push_back(pred_label);

            if (pred_label == images[i].label) correct++;

            if ((i + 1) % 50 == 0 || i == images.size() - 1) {
                std::cout << "Processed " << (i + 1) << "/" << images.size()
                    << " | Running accuracy: "
                    << (100.0 * correct / (i + 1)) << "%" << std::endl;
            }
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double total_sec = std::chrono::duration<double>(end_time - start_time).count();

        std::cout << "\n===== Inference Complete =====" << std::endl;
        std::cout << "Total images:  " << images.size() << std::endl;
        std::cout << "Correct:       " << correct << std::endl;
        std::cout << "Accuracy:      " << (100.0 * correct / images.size()) << "%" << std::endl;
        std::cout << "Total time:    " << total_sec << " sec" << std::endl;
        std::cout << "Per image:     " << (total_sec / images.size()) << " sec" << std::endl;

        // ---- Dina's part: MetricsComputer + Visualizer ----

        // compute precision, recall, f1, confusion matrix
        Metrics m = MetricsComputer::compute(predicted_labels, true_labels);
        MetricsComputer::printReport(m);
        MetricsComputer::printConfusionMatrix(m);

        // save 10 annotated x-ray images to output/ folder for the report
        Visualizer viz("output");
        int saved = 0;

        for (size_t i = 0; i < images.size() && saved < 10; i++) {
            auto tensor = ImagePreprocessor::preprocess(images[i].image);
            auto logits = model.infer(tensor);

            // softmax to get confidence score for pneumonia
            float exp0 = std::exp(logits[0]);
            float exp1 = std::exp(logits[1]);
            float conf = exp1 / (exp0 + exp1);

            std::string pred = (conf >= threshold) ? "PNEUMONIA" : "NORMAL";
            viz.annotateAndSave(images[i].image, images[i].filename, pred, conf);
            saved++;
        }

    }
    catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}