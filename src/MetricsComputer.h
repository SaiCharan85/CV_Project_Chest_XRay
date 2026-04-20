#pragma once
#include <vector>

struct Metrics {
    float accuracy;
    float precision;
    float recall;
    float f1;
    int tp;
    int tn;
    int fp;
    int fn;
};

class MetricsComputer {
public:
    static Metrics compute(const std::vector<int>& predicted,
        const std::vector<int>& true_labels);
    static void printReport(const Metrics& m);
    static void printConfusionMatrix(const Metrics& m);
};