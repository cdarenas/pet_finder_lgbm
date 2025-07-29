#include "metrics.hpp"
#include <iostream>
#include <vector>
#include <algorithm>

// Accuracy simple
double accuracy(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
	int correct = 0;
	for (size_t i = 0; i < y_true.size(); ++i)
		if (y_true[i] == y_pred[i]) correct++;
	return static_cast<double>(correct) / y_true.size();
}

// F1 macro
double f1_score_macro(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
	const int NUM_CLASSES = 5;
	std::vector<int> tp(NUM_CLASSES, 0), fp(NUM_CLASSES, 0), fn(NUM_CLASSES, 0);

	for (size_t i = 0; i < y_true.size(); ++i) {
		int true_label = y_true[i];
		int pred_label = y_pred[i];
		if (true_label == pred_label)
			tp[true_label]++;
		else {
			fp[pred_label]++;
			fn[true_label]++;
		}
	}

	double f1_total = 0.0;
	for (int i = 0; i < NUM_CLASSES; ++i) {
		double precision = tp[i] + fp[i] > 0 ? (double)tp[i] / (tp[i] + fp[i]) : 0.0;
		double recall = tp[i] + fn[i] > 0 ? (double)tp[i] / (tp[i] + fn[i]) : 0.0;
		double f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0;
		f1_total += f1;
	}
	return f1_total / NUM_CLASSES;
}

// Cohen's Kappa
double cohen_kappa(const std::vector<int>& y_true, const std::vector<int>& y_pred, int num_classes) {
	if (y_true.size() != y_pred.size() || y_true.empty()) return 0.0;

	std::vector<std::vector<int>> confusion(num_classes, std::vector<int>(num_classes, 0));
	std::vector<int> row_totals(num_classes, 0), col_totals(num_classes, 0);
	int total = 0;

	for (size_t i = 0; i < y_true.size(); ++i) {
		int true_label = y_true[i];
		int pred_label = y_pred[i];

		// Validar que estén en rango
		if (true_label < 0 || true_label >= num_classes || pred_label < 0 || pred_label >= num_classes)
			continue;

		confusion[true_label][pred_label]++;
		row_totals[true_label]++;
		col_totals[pred_label]++;
		total++;
	}

	if (total == 0) return 0.0;

	// Acuerdo observado (Po)
	double po = 0.0;
	for (int i = 0; i < num_classes; ++i)
		po += confusion[i][i];
	po /= total;

	// Acuerdo esperado (Pe)
	double pe = 0.0;
	for (int i = 0; i < num_classes; ++i)
		pe += static_cast<double>(row_totals[i]) * col_totals[i];
	pe /= (total * total);

	return (pe < 1.0) ? (po - pe) / (1.0 - pe) : 0.0;
}

// Imprimir matriz de confusión
void print_confusion_matrix(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
	const int NUM_CLASSES = 5;
	std::vector<std::vector<int>> matrix(NUM_CLASSES, std::vector<int>(NUM_CLASSES, 0));
	for (size_t i = 0; i < y_true.size(); ++i) {
		matrix[y_true[i]][y_pred[i]]++;
	}
	std::cout << "\nMatriz de confusion:\n";
	for (int i = 0; i < NUM_CLASSES; ++i) {
		for (int j = 0; j < NUM_CLASSES; ++j) {
			std::cout << matrix[i][j] << "\t";
		}
		std::cout << std::endl;
	}
}
