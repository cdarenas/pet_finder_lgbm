#include "io_utils.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

using namespace std;

// Leer matriz de predicciones (una fila por observación, con probabilidades)
vector<int> read_predicted_classes(const string& filename) {
	ifstream file(filename);
	vector<int> predicted_classes;
	string line;
	while (getline(file, line)) {
		stringstream ss(line);
		double prob;
		vector<double> probs;
		while (ss >> prob) {
			probs.push_back(prob);
		}
		int pred_class = static_cast<int>(max_element(probs.begin(), probs.end()) - probs.begin());
		predicted_classes.push_back(pred_class);
	}
	return predicted_classes;
}

// Leer etiquetas reales desde archivo
vector<int> read_labels(const string& filename) {
	ifstream file(filename);
	vector<int> labels;
	int val;
	while (file >> val) labels.push_back(val);
	return labels;
}

// Lee archivo de configuración
string read_config(const string& filename) {
	ifstream file(filename);
	stringstream buffer;
	buffer << file.rdbuf();
	return buffer.str();
}

// Guarda datos de vector en un archivo CSV
void save_vector_to_csv(const string& filename, const vector<int>& data) {
	ofstream file(filename);
	for (const auto& val : data) {
		file << val << "\n";
	}
}

// Almacena en un solo archivo CSV las etiquetas reales y las predicciones
void save_combined_csv(const string& filename, const vector<int>& y_true, const vector<int>& y_pred) {
	ofstream file(filename);
	file << "indice,y_true,y_pred\n";
	for (size_t i = 0; i < y_true.size(); ++i) {
		file << i << "," << y_true[i] << "," << y_pred[i] << "\n";
	}
}
