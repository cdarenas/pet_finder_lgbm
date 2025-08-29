#pragma once
#include <vector>

double accuracy(const std::vector<int>& y_true, const std::vector<int>& y_pred);

double f1_score_macro(const std::vector<int>& y_true, const std::vector<int>& y_pred);

double cohen_kappa(const std::vector<int>& y_true, const std::vector<int>& y_pred, int num_classes = 5);

double quadratic_weighted_kappa(const std::vector<int>& y_true, const std::vector<int>& y_pred, int num_classes = 5);

void print_confusion_matrix(const std::vector<int>& y_true, const std::vector<int>& y_pred);
