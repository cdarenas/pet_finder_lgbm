#ifndef DATABASE_HPP
#define DATABASE_HPP

#include <string>
#include <vector>

int insert_result_sqlite(double acc, double f1, double kappa, const std::string& model_path,
	const std::string& conf_path, const std::string& config_str);

void insert_predictions_sqlite(const std::vector<int>& y_true,
	const std::vector<int>& y_pred,
	int result_id);

void save_best_model(const std::string& db_path = "resultados.db");

void save_best_model_by_kappa();

void save_final_model(const std::string& db_path, const std::string& model_path, const std::string& config_path);

#endif // DATABASE_HPP
