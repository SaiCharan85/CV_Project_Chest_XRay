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

        // Store per-image results for visualization
        std::vector<float> confidences;
        std::vector<std::string> predictions;

        int correct = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < images.size(); i++) {
            auto tensor = ImagePreprocessor::preprocess(images[i].image);

            // get raw logits and compute softmax
            auto logits = model.infer(tensor);
            float exp0 = std::exp(logits[0]);
            float exp1 = std::exp(logits[1]);
            float conf = exp1 / (exp0 + exp1);

            std::string prediction = (conf >= threshold) ? "PNEUMONIA" : "NORMAL";
            int pred_label = (prediction == "PNEUMONIA") ? 1 : 0;

            true_labels.push_back(images[i].label);
            predicted_labels.push_back(pred_label);
            confidences.push_back(conf);
            predictions.push_back(prediction);

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

        // save annotated x-ray images to output/ folder for the report
        // collect images into 4 buckets for a balanced visualization
        Visualizer viz("output");

        // TP = actual pneumonia, predicted pneumonia (correct)
        // TN = actual normal,    predicted normal    (correct)
        // FP = actual normal,    predicted pneumonia (wrong)
        // FN = actual pneumonia, predicted normal    (wrong)
        std::vector<size_t> tp_indices, tn_indices, fp_indices, fn_indices;

        for (size_t i = 0; i < images.size(); i++) {
            if (true_labels[i] == 1 && predicted_labels[i] == 1)      tp_indices.push_back(i);
            else if (true_labels[i] == 0 && predicted_labels[i] == 0) tn_indices.push_back(i);
            else if (true_labels[i] == 0 && predicted_labels[i] == 1) fp_indices.push_back(i);
            else if (true_labels[i] == 1 && predicted_labels[i] == 0) fn_indices.push_back(i);
        }

        std::cout << "\nVisualization buckets:"
            << "  TP=" << tp_indices.size()
            << "  TN=" << tn_indices.size()
            << "  FP=" << fp_indices.size()
            << "  FN=" << fn_indices.size() << std::endl;

        // save ~3 from each bucket (12 total, or fewer if a bucket is small)
        auto saveFromBucket = [&](const std::vector<size_t>& bucket,
            int count, const std::string& label) {
                int n = std::min(count, (int)bucket.size());
                for (int i = 0; i < n; i++) {
                    size_t idx = bucket[i];
                    std::string actual = (images[idx].label == 1) ? "PNEUMONIA" : "NORMAL";
                    viz.annotateAndSave(images[idx].image, images[idx].filename,
                        actual, predictions[idx], confidences[idx]);
                }
                std::cout << "Saved " << n << " " << label << " images" << std::endl;
                return n;
            };

        int saved = 0;
        saved += saveFromBucket(tp_indices, 3, "True Positive  (actual PNEUMONIA, pred PNEUMONIA)");
        saved += saveFromBucket(tn_indices, 3, "True Negative  (actual NORMAL,    pred NORMAL)");
        saved += saveFromBucket(fp_indices, 3, "False Positive (actual NORMAL,    pred PNEUMONIA)");
        saved += saveFromBucket(fn_indices, 3, "False Negative (actual PNEUMONIA, pred NORMAL)");

        std::cout << "Total annotated images saved: " << saved << std::endl;

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