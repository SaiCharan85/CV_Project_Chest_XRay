#include "MetricsComputer.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>

// computes accuracy, precision, recall, f1 and confusion matrix values
Metrics MetricsComputer::compute(const std::vector<int>& predicted,
                                  const std::vector<int>& true_labels) {

    // make sure both vectors are the same length
    if (predicted.size() != true_labels.size())
        throw std::runtime_error("predicted and true_labels size don't match");

    int tp = 0, tn = 0, fp = 0, fn = 0;

    // go through each prediction and count tp, tn, fp, fn
    for (size_t i = 0; i < predicted.size(); i++) {
        int p = predicted[i];
        int t = true_labels[i];

        if (p == 1 && t == 1) tp++;       // correctly predicted pneumonia
        else if (p == 0 && t == 0) tn++;  // correctly predicted normal
        else if (p == 1 && t == 0) fp++;  // predicted pneumonia but was normal
        else if (p == 0 && t == 1) fn++;  // predicted normal but was pneumonia
    }

    int total = tp + tn + fp + fn;

    // standard metric formulas
    float accuracy  = (float)(tp + tn) / total;
    float precision = (tp + fp) > 0 ? (float)tp / (tp + fp) : 0.0f;
    float recall    = (tp + fn) > 0 ? (float)tp / (tp + fn) : 0.0f;
    float f1        = (precision + recall) > 0
                        ? 2.0f * precision * recall / (precision + recall)
                        : 0.0f;

    return {accuracy, precision, recall, f1, tp, tn, fp, fn};
}

// prints the evaluation metrics to the terminal
void MetricsComputer::printReport(const Metrics& m) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n========== Evaluation Report ==========\n";
    std::cout << "  Accuracy  : " << m.accuracy  << "\n";
    std::cout << "  Precision : " << m.precision << "\n";
    std::cout << "  Recall    : " << m.recall    << "\n";
    std::cout << "  F1 Score  : " << m.f1        << "\n";
    std::cout << "========================================\n";
}

// prints a 2x2 confusion matrix to the terminal
void MetricsComputer::printConfusionMatrix(const Metrics& m) {
    std::cout << "\n========= Confusion Matrix =============\n";
    std::cout << "                Predicted\n";
    std::cout << "              NORMAL  PNEUMONIA\n";
    std::cout << "Actual NORMAL  [" << std::setw(4) << m.tn << "     " << std::setw(4) << m.fp << " ]\n";
    std::cout << "Actual PNEUM   [" << std::setw(4) << m.fn << "     " << std::setw(4) << m.tp << " ]\n";
    std::cout << "========================================\n";
}