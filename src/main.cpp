/*
===============================================================
	PetFinderLGBM - Clasificación de Velocidad de Adopción
	Autor: Cristian Arenas
	Descripción:
		- Entrenamiento y evaluación de modelos LightGBM
		- Validación cruzada con K-Fold
		- Métricas avanzadas: Accuracy, F1 macro, Kappa
		- Almacenamiento de resultados en SQLite3
		- Visualización de resultados con Python
*/

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <map>
#include <algorithm>
#include <numeric>
#include <sqlite3.h>
#include <ctime>
#include <filesystem>
#include "io_utils.hpp"
#include "metrics.hpp"
#include "database.hpp"
#include <windows.h>

// Códigos ANSI para color
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define BOLD    "\033[1m"

using namespace std;
namespace fs = std::filesystem;

const int NUM_CLASSES = 5;
const int NUM_FOLDS = 5;

int main() {
	char result_path[MAX_PATH];
	GetModuleFileNameA(NULL, result_path, MAX_PATH);
	fs::path exe_path = fs::path(result_path).parent_path();

	fs::path fold_dir = exe_path / "folds";
	fs::path lightgbm_path = exe_path / "lightgbm.exe";

	double total_acc = 0.0, total_f1 = 0.0;
	double total_kappa = 0.0;

	// Archivo global para guardar todas las predicciones
	ofstream global_csv(exe_path / "predicciones_completas.csv");
	global_csv << "fold,indice,y_true,y_pred\n";

	for (int fold = 0; fold < NUM_FOLDS; ++fold) {
		cout << "\n=== Fold " << fold << " ===" << endl;

		string train_file = (fold_dir / ("train_fold_" + to_string(fold) + ".txt")).string();
		string valid_file = (fold_dir / ("valid_fold_" + to_string(fold) + ".txt")).string();
		string valid_labels = (fold_dir / ("y_valid_fold_" + to_string(fold) + ".txt")).string();
		string model_file = (fold_dir / ("model_fold_" + to_string(fold) + ".txt")).string();
		string pred_file = (fold_dir / ("predictions_fold_" + to_string(fold) + ".txt")).string();
		string config_train = (fold_dir / ("config_train_fold_" + to_string(fold) + ".txt")).string();
		string config_pred = (fold_dir / ("config_pred_fold_" + to_string(fold) + ".txt")).string();

		if (system((lightgbm_path.string() + " config=" + config_train).c_str()) != 0) {
			cerr << RED << BOLD << "Error en entrenamiento del fold " << fold << endl;
			continue;
		}

		if (system((lightgbm_path.string() + " config=" + config_pred).c_str()) != 0) {
			cerr << RED << BOLD << "Error en prediccion del fold " << fold << endl;
			continue;
		}

		// Leer resultados
		vector<int> y_true = read_labels(valid_labels);
		vector<int> y_pred = read_predicted_classes(pred_file);

		if (y_true.size() != y_pred.size()) {
			cerr << RED << BOLD << "Tamaño inconsistente en fold " << fold << endl;
			continue;
		}

		double acc = accuracy(y_true, y_pred);
		double f1 = f1_score_macro(y_true, y_pred);
		total_acc += acc;
		total_f1 += f1;
		double kappa = cohen_kappa(y_true, y_pred);
		total_kappa += kappa;

		cout << GREEN << BOLD << "Fold " << fold << " - Accuracy: " << acc << ", F1 macro: " << f1 << ", Kappa: " << kappa << RESET << endl;

		// Calcular y mostrar matriz de confusión
		vector<vector<int>> matrix(NUM_CLASSES, vector<int>(NUM_CLASSES, 0));
		for (size_t i = 0; i < y_true.size(); ++i) {
			matrix[y_true[i]][y_pred[i]]++;
		}

		cout << "\nMatriz de confusion fold " << fold << ":" << endl;
		for (int i = 0; i < NUM_CLASSES; ++i) {
			for (int j = 0; j < NUM_CLASSES; ++j) {
				cout << matrix[i][j] << "\t";
			}
			cout << endl;
		}

		// Guardar matriz de confusión como CSV
		ofstream matrix_file((exe_path / ("matriz_confusion_fold_" + to_string(fold) + ".csv")).string());
		for (int i = 0; i < NUM_CLASSES; ++i) {
			for (int j = 0; j < NUM_CLASSES; ++j) {
				matrix_file << matrix[i][j];
				if (j < NUM_CLASSES - 1) matrix_file << ",";
			}
			matrix_file << "\n";
		}
		matrix_file.close();

		// Guardar resultados
		string conf_str = read_config(config_train);
		int result_id = insert_result_sqlite(acc, f1, kappa, model_file, config_train, conf_str);
		if (result_id != -1)
			insert_predictions_sqlite(y_true, y_pred, result_id);

		// Guardar y_true y y_pred como CSV individuales
		string y_true_csv = (exe_path / ("y_true_fold_" + to_string(fold) + ".csv")).string();
		string y_pred_csv = (exe_path / ("y_pred_fold_" + to_string(fold) + ".csv")).string();
		save_vector_to_csv(y_true_csv, y_true);
		save_vector_to_csv(y_pred_csv, y_pred);
		// Guardar combinados en un CSV
		save_combined_csv((exe_path / ("y_pred_vs_true_fold_" + to_string(fold) + ".csv")).string(), y_true, y_pred);

		// Generar gráfico de matriz de confusión
		fs::path output_img = exe_path / ("conf_matrix_fold_" + to_string(fold) + ".png");
		fs::path script_plot = exe_path / "scripts" / "plot_confusion_matrix.py";
		string python_cmd = "python \"" + script_plot.string() + "\" \"" + y_true_csv + "\" \"" + y_pred_csv + "\" \"" + output_img.string() + "\"";
		cout << "Generando matriz de confusion para fold " << fold << "..." << endl;
		system(python_cmd.c_str());

		// Agregar al CSV global las predicciones de este fold
		for (size_t i = 0; i < y_true.size(); ++i) {
			global_csv << fold << "," << i << "," << y_true[i] << "," << y_pred[i] << "\n";
		}
	}

	global_csv.close();

	save_best_model();
	save_best_model_by_kappa();

	cout << CYAN << BOLD << "\n==== RESULTADO PROMEDIO ====" << endl;
	cout << YELLOW << "Accuracy promedio: " << (total_acc / NUM_FOLDS) << endl;
	cout << YELLOW << "F1 macro promedio: " << (total_f1 / NUM_FOLDS) << endl;
	cout << YELLOW << "Kappa promedio: " << (total_kappa / NUM_FOLDS) << RESET << endl;

	// Scripts de análisis visual
	fs::path analysis1 = exe_path / "scripts" / "analysis_results.py";
	fs::path analysis2 = exe_path / "scripts" / "analysis_results2.py";

	cout << "\nEjecutando analisis visual en Python..." << endl;
	system(("python \"" + analysis1.string() + "\"").c_str());

	cout << "\nEjecutando analisis visual completo en Python..." << endl;
	system(("python \"" + analysis2.string() + "\"").c_str());

	// Generar gráfico de importancia de variables
	fs::path script_importance = exe_path / "scripts" / "analyze_feature_importance.py";
	fs::path output_importance_img = exe_path / "importancia_variables.png";
	string importance_cmd = "python \"" + script_importance.string() + "\" \"" + fold_dir.string() + "\" \"" + output_importance_img.string() + "\"";
	cout << "\nGenerando grafico de importancia de variables..." << endl;
	system(importance_cmd.c_str());

	char user_input;
	cout << YELLOW << "\n¿Deseas continuar con el entrenamiento final del modelo con todo el dataset? (S/N): " << RESET;
	cin >> user_input;

	if (toupper(user_input) != 'S') {
		cout << RED << "❌ Proceso de entrenamiento final cancelado por el usuario." << RESET << endl;
		return 0;
	}

	// --- Entrenamiento final con todo el dataset ---
	cout << BLUE << BOLD << "\n=== 🚀 Entrenando modelo final con todo el dataset ===" << RESET << endl;

	string train_all_file = (fold_dir / "train_all.txt").string();
	string config_final_file = (fold_dir / "config_train_all.txt").string();
	string final_model_file = (fold_dir / "final_model.txt").string();

	string final_cmd = lightgbm_path.string() + " config=" + config_final_file;
	if (system(final_cmd.c_str()) == 0) {
		cout << GREEN << BOLD << "✅ Modelo final entrenado correctamente: " << final_model_file << RESET << endl;
	}
	else {
		cerr << RED << BOLD << "❌ Error al entrenar el modelo final." << RESET << endl;
	}

	fs::path final_model = exe_path / "folds" / "model_all.txt";
	fs::path final_config = exe_path / "folds" / "config_train_all.txt";
	save_final_model("resultados.db", final_model.string(), final_config.string());

	// -----------------EVALUACIÓN FINAL DEL MODELO-----------------
	cout << GREEN << "\nEvaluando modelo final sobre todos los datos..." << RESET << endl;

	fs::path model_final = exe_path / "modelo_final.txt";
	string pred_final = (fold_dir / "predictions_final.txt").string();
	string config_pred_final = (fold_dir / "config_pred_final.txt").string();

	// Crear archivo de predicción para el modelo final
	ofstream fcfg(config_pred_final);
	fcfg << "task=predict\n";
	fcfg << "input_model=" << model_final.string() << "\n";
	fcfg << "data=" << train_all_file << "\n";
	fcfg << "output_result=" << pred_final << "\n";
	fcfg.close();

	// Ejecutar predicción
	if (system((lightgbm_path.string() + " config=" + config_pred_final).c_str()) != 0) {
		cerr << RED << BOLD << "❌ Error en prediccion con modelo final." << RESET << endl;
	}
	else {
		cout << RED << BOLD << "✅ Prediccion con modelo final realizada." << RESET << endl;

		// Leer etiquetas verdaderas (de train_all)
		vector<int> y_true_final;
		ifstream fin(train_all_file);
		string line;
		while (getline(fin, line)) {
			stringstream ss(line);
			int label;
			ss >> label;
			y_true_final.push_back(label);
		}

		// Leer predicciones
		vector<int> y_pred_final = read_predicted_classes(pred_final);

		// Calcular métricas
		double acc_final = accuracy(y_true_final, y_pred_final);
		double f1_final = f1_score_macro(y_true_final, y_pred_final);
		double kappa_final = cohen_kappa(y_true_final, y_pred_final);

		cout << CYAN << BOLD << "\n📊 Metricas del modelo final sobre todo el dataset:" << endl;
		cout << YELLOW << " - Accuracy: " << acc_final << endl;
		cout << YELLOW << " - F1 macro: " << f1_final << endl;
		cout << YELLOW << " - Kappa: " << kappa_final << RESET << endl;

		// Matriz de confusión
		vector<vector<int>> matrix(NUM_CLASSES, vector<int>(NUM_CLASSES, 0));
		for (size_t i = 0; i < y_true_final.size(); ++i) {
			matrix[y_true_final[i]][y_pred_final[i]]++;
		}
		cout << "\n🧩 Matriz de confusion del modelo final:" << endl;
		for (int i = 0; i < NUM_CLASSES; ++i) {
			for (int j = 0; j < NUM_CLASSES; ++j) {
				cout << matrix[i][j] << "\t";
			}
			cout << endl;
		}

		// Guardar archivos
		string final_matrix_csv = (exe_path / "matriz_confusion_final.csv").string();
		string y_true_final_csv = (exe_path / "y_true_final.csv").string();
		string y_pred_final_csv = (exe_path / "y_pred_final.csv").string();
		string y_pred_vs_true = (exe_path / "y_pred_vs_true_final.csv").string();
		save_vector_to_csv(y_true_final_csv, y_true_final);
		save_vector_to_csv(y_pred_final_csv, y_pred_final);
		save_combined_csv(y_pred_vs_true, y_true_final, y_pred_final);

		ofstream fout(final_matrix_csv);
		for (int i = 0; i < NUM_CLASSES; ++i) {
			for (int j = 0; j < NUM_CLASSES; ++j) {
				fout << matrix[i][j];
				if (j < NUM_CLASSES - 1) fout << ",";
			}
			fout << "\n";
		}
		fout.close();

		// Generar imagen de matriz
		fs::path script_plot = exe_path / "scripts" / "plot_confusion_matrix.py";
		fs::path output_img = exe_path / "conf_matrix_final.png";
		string python_cmd = "python \"" + script_plot.string() + "\" \"" + y_true_final_csv + "\" \"" + y_pred_final_csv + "\" \"" + output_img.string() + "\"";
		system(python_cmd.c_str());

		// Insertar en SQLite
		string conf_str = read_config(config_final_file);
		int result_id = insert_result_sqlite(acc_final, f1_final, kappa_final, model_final.string(), config_final_file, conf_str);
		if (result_id != -1) {
			insert_predictions_sqlite(y_true_final, y_pred_final, result_id);
			save_best_model();  // Marca como modelo final
		}
	}

	return 0;
}

