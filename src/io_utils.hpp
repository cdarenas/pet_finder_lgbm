// Archivo: io_utils.hpp
#pragma once
#include <string>
#include <vector>

std::vector<int> read_labels(const std::string& filename);

std::vector<int> read_predicted_classes(const std::string& filename);

std::string read_config(const std::string& filename);

void save_vector_to_csv(const std::string& filename, const std::vector<int>& data);

void save_combined_csv(const std::string& filename, const std::vector<int>& y_true, const std::vector<int>& y_pred);

