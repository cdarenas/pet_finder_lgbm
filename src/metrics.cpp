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

// Quadratic Weighted Kappa
double quadratic_weighted_kappa(const std::vector<int>& y_true, const std::vector<int>& y_pred, int num_classes) {
	if (y_true.size() != y_pred.size() || y_true.empty()) return 0.0;

	const int N = static_cast<int>(y_true.size());

	// Matriz de confusión O (observada)
	std::vector<std::vector<double>> O(num_classes, std::vector<double>(num_classes, 0.0));
	// Histogramas por clase
	std::vector<double> hist_true(num_classes, 0.0), hist_pred(num_classes, 0.0);

	for (size_t i = 0; i < y_true.size(); ++i) {
		int t = y_true[i];
		int p = y_pred[i];
		if (t >= 0 && t < num_classes && p >= 0 && p < num_classes) {
			O[t][p] += 1.0;
			hist_true[t] += 1.0;
			hist_pred[p] += 1.0;
		}
	}

	// Matriz de pesos W_ij = (i - j)^2 / (K - 1)^2
	const double denom_w = (num_classes > 1) ? ((num_classes - 1.0) * (num_classes - 1.0)) : 1.0;
	std::vector<std::vector<double>> W(num_classes, std::vector<double>(num_classes, 0.0));
	for (int i = 0; i < num_classes; ++i) {
		for (int j = 0; j < num_classes; ++j) {
			double diff = static_cast<double>(i - j);
			W[i][j] = (diff * diff) / denom_w;
		}
	}

	// Matriz esperada E = outer(hist_true, hist_pred) / N
	std::vector<std::vector<double>> E(num_classes, std::vector<double>(num_classes, 0.0));
	for (int i = 0; i < num_classes; ++i) {
		for (int j = 0; j < num_classes; ++j) {
			E[i][j] = (hist_true[i] * hist_pred[j]) / static_cast<double>(N);
		}
	}

	// Sumas ponderadas
	double sum_w_o = 0.0;
	double sum_w_e = 0.0;
	for (int i = 0; i < num_classes; ++i) {
		for (int j = 0; j < num_classes; ++j) {
			sum_w_o += W[i][j] * O[i][j];
			sum_w_e += W[i][j] * E[i][j];
		}
	}

	if (sum_w_e <= 0.0) {
		// Sin variación esperada; evita división por cero
		return 0.0;
	}

	// QWK = 1 - (sum(W * O) / sum(W * E))
	double kappa = 1.0 - (sum_w_o / sum_w_e);
	return kappa;
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
