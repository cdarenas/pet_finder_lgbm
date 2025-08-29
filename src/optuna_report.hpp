#pragma once
#include <filesystem>

// Genera archivos de reporte a partir de folds/optuna_study.db:
// - optuna_history_<study>.csv
// - optuna_best_params_<study>.json
// - optuna_curve_best_<study>.csv
//
// folds_dir: carpeta que contiene optuna_study.db (normalmente .../folds)
// out_dir  : carpeta de salida para los reportes (por ejemplo exe_path)
//
// Devuelve true si pudo abrir la base y procesar al menos 1 estudio.
bool generate_optuna_report(const std::filesystem::path& folds_dir,
	const std::filesystem::path& out_dir);
