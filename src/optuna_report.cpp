#include "optuna_report.hpp"

#include <sqlite3.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <limits>
#include <utility>

namespace fs = std::filesystem;

// Colores opcionales: si el binario principal ya los define, no se re-definen
#ifndef RESET
#define RESET   ""
#define RED     ""
#define GREEN   ""
#define YELLOW  ""
#define BLUE    ""
#define CYAN    ""
#define BOLD    ""
#endif

namespace {
	bool table_exists(sqlite3* db, const std::string& name) {
		sqlite3_stmt* stmt = nullptr;
		bool exists = false;
		const char* q = "SELECT name FROM sqlite_master WHERE type='table' AND name=? LIMIT 1";
		if (sqlite3_prepare_v2(db, q, -1, &stmt, nullptr) == SQLITE_OK) {
			sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT);
			exists = (sqlite3_step(stmt) == SQLITE_ROW);
		}
		if (stmt) sqlite3_finalize(stmt);
		return exists;
	}

	struct StudyInfo { int id; std::string name; };
	std::vector<StudyInfo> get_studies(sqlite3* db) {
		std::vector<StudyInfo> out;
		sqlite3_stmt* stmt = nullptr;
		const char* q = "SELECT study_id, study_name FROM studies ORDER BY study_id";
		if (sqlite3_prepare_v2(db, q, -1, &stmt, nullptr) == SQLITE_OK) {
			while (sqlite3_step(stmt) == SQLITE_ROW) {
				StudyInfo s;
				s.id = sqlite3_column_int(stmt, 0);
				const unsigned char* txt = sqlite3_column_text(stmt, 1);
				s.name = txt ? reinterpret_cast<const char*>(txt) : "";
				out.push_back(std::move(s));
			}
		}
		if (stmt) sqlite3_finalize(stmt);
		return out;
	}

	// 1 = maximize, 0 = minimize (fallback maximize)
	int get_study_direction(sqlite3* db, int study_id, int default_dir = 1) {
		if (table_exists(db, "study_directions")) {
			sqlite3_stmt* stmt = nullptr;
			const char* q = "SELECT direction FROM study_directions WHERE study_id=? LIMIT 1";
			if (sqlite3_prepare_v2(db, q, -1, &stmt, nullptr) == SQLITE_OK) {
				sqlite3_bind_int(stmt, 1, study_id);
				if (sqlite3_step(stmt) == SQLITE_ROW) {
					int dir = sqlite3_column_int(stmt, 0);
					sqlite3_finalize(stmt);
					return dir;
				}
				sqlite3_finalize(stmt);
			}
		}
		return default_dir;
	}

	struct BestTrial {
		int trial_id = -1;
		int number = -1;
		double value = std::numeric_limits<double>::quiet_NaN();
	};

	BestTrial get_best_trial(sqlite3* db, int study_id, int direction /*0=min,1=max*/) {
		BestTrial bt;
		std::string order = (direction == 1 ? "DESC" : "ASC");
		std::string q =
			"SELECT t.trial_id, t.number, v.value "
			"FROM trials t "
			"JOIN trial_values v ON v.trial_id = t.trial_id "
			"WHERE t.study_id=? AND t.state=1 "
			"ORDER BY v.value " + order + " "
			"LIMIT 1";
		sqlite3_stmt* stmt = nullptr;
		if (sqlite3_prepare_v2(db, q.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
			sqlite3_bind_int(stmt, 1, study_id);
			if (sqlite3_step(stmt) == SQLITE_ROW) {
				bt.trial_id = sqlite3_column_int(stmt, 0);
				bt.number = sqlite3_column_int(stmt, 1);
				bt.value = sqlite3_column_double(stmt, 2);
			}
		}
		if (stmt) sqlite3_finalize(stmt);
		return bt;
	}

	std::vector<std::pair<std::string, std::string>> get_trial_params(sqlite3* db, int trial_id) {
		std::vector<std::pair<std::string, std::string>> params;
		sqlite3_stmt* stmt = nullptr;
		const char* q = "SELECT param_name, param_value FROM trial_params WHERE trial_id=? ORDER BY param_name";
		if (sqlite3_prepare_v2(db, q, -1, &stmt, nullptr) == SQLITE_OK) {
			sqlite3_bind_int(stmt, 1, trial_id);
			while (sqlite3_step(stmt) == SQLITE_ROW) {
				const unsigned char* n = sqlite3_column_text(stmt, 0);
				const unsigned char* v = sqlite3_column_text(stmt, 1);
				params.emplace_back(
					n ? reinterpret_cast<const char*>(n) : "",
					v ? reinterpret_cast<const char*>(v) : ""
				);
			}
		}
		if (stmt) sqlite3_finalize(stmt);
		return params;
	}

	void write_history_csv(sqlite3* db, int study_id, const fs::path& out_csv) {
		std::ofstream out(out_csv);
		out << "trial_number,value,state,datetime_start,datetime_complete\n";
		const char* q =
			"SELECT t.number, v.value, t.state, t.datetime_start, t.datetime_complete "
			"FROM trials t "
			"LEFT JOIN trial_values v ON v.trial_id = t.trial_id "
			"WHERE t.study_id=? "
			"ORDER BY t.number";
		sqlite3_stmt* stmt = nullptr;
		if (sqlite3_prepare_v2(db, q, -1, &stmt, nullptr) == SQLITE_OK) {
			sqlite3_bind_int(stmt, 1, study_id);
			while (sqlite3_step(stmt) == SQLITE_ROW) {
				int number = sqlite3_column_int(stmt, 0);
				double value = (sqlite3_column_type(stmt, 1) == SQLITE_NULL)
					? std::numeric_limits<double>::quiet_NaN()
					: sqlite3_column_double(stmt, 1);
				int state = sqlite3_column_int(stmt, 2);
				const unsigned char* ds = sqlite3_column_text(stmt, 3);
				const unsigned char* dc = sqlite3_column_text(stmt, 4);

				out << number << ",";
				if (std::isnan(value)) out << ""; else out << value;
				out << "," << state << ","
					<< (ds ? reinterpret_cast<const char*>(ds) : "") << ","
					<< (dc ? reinterpret_cast<const char*>(dc) : "") << "\n";
			}
		}
		if (stmt) sqlite3_finalize(stmt);
	}

	void write_intermediate_csv(sqlite3* db, int trial_id, const fs::path& out_csv) {
		if (!table_exists(db, "trial_intermediate_values")) return;
		std::ofstream out(out_csv);
		out << "step,value\n";
		sqlite3_stmt* stmt = nullptr;
		const char* q = "SELECT step, intermediate_value FROM trial_intermediate_values WHERE trial_id=? ORDER BY step";
		if (sqlite3_prepare_v2(db, q, -1, &stmt, nullptr) == SQLITE_OK) {
			sqlite3_bind_int(stmt, 1, trial_id);
			while (sqlite3_step(stmt) == SQLITE_ROW) {
				int step = sqlite3_column_int(stmt, 0);
				double val = sqlite3_column_double(stmt, 1);
				out << step << "," << val << "\n";
			}
		}
		if (stmt) sqlite3_finalize(stmt);
	}

	std::string json_escape(const std::string& s) {
		std::ostringstream o;
		for (char c : s) {
			switch (c) {
			case '\"': o << "\\\""; break;
			case '\\': o << "\\\\"; break;
			case '\b': o << "\\b"; break;
			case '\f': o << "\\f"; break;
			case '\n': o << "\\n"; break;
			case '\r': o << "\\r"; break;
			case '\t': o << "\\t"; break;
			default:
				if (static_cast<unsigned char>(c) < 0x20) {
					o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
				}
				else o << c;
			}
		}
		return o.str();
	}

	void write_best_params_json(const std::vector<std::pair<std::string, std::string>>& params,
		int study_id, int trial_number, double best_value,
		int direction, const fs::path& out_json) {
		std::ofstream out(out_json);
		out << "{\n";
		out << "  \"study_id\": " << study_id << ",\n";
		out << "  \"trial_number\": " << trial_number << ",\n";
		out << "  \"direction\": \"" << (direction == 1 ? "maximize" : "minimize") << "\",\n";
		out << "  \"best_value\": " << best_value << ",\n";
		out << "  \"params\": {\n";
		for (size_t i = 0; i < params.size(); ++i) {
			out << "    \"" << json_escape(params[i].first) << "\": \"" << json_escape(params[i].second) << "\"";
			out << (i + 1 < params.size() ? ",\n" : "\n");
		}
		out << "  }\n";
		out << "}\n";
	}
} // namespace

bool generate_optuna_report(const fs::path& folds_dir, const fs::path& out_dir) {
	fs::path db_path = folds_dir / "optuna_study.db";
	if (!fs::exists(db_path)) {
		std::cerr << YELLOW << "[WARN] No se encontró " << db_path << ". Copiá la carpeta 'folds' del Python." << RESET << std::endl;
		return false;
	}

	sqlite3* db = nullptr;
	if (sqlite3_open(db_path.string().c_str(), &db) != SQLITE_OK) {
		std::cerr << RED << "[ERROR] No se pudo abrir SQLite: " << db_path << RESET << std::endl;
		return false;
	}

	auto studies = get_studies(db);
	if (studies.empty()) {
		std::cerr << YELLOW << "[WARN] Base de Optuna sin estudios." << RESET << std::endl;
		sqlite3_close(db);
		return false;
	}

	std::cout << CYAN << BOLD << "\n=== Reporte Optuna (desde " << db_path.filename().string() << ") ===" << RESET << std::endl;

	bool any = false;
	for (const auto& s : studies) {
		int study_id = s.id;
		std::string study_name = s.name;
		int dir = get_study_direction(db, study_id, 1);
		BestTrial bt = get_best_trial(db, study_id, dir);

		fs::path hist_csv = out_dir / ("optuna_history_" + study_name + ".csv");
		fs::path best_json = out_dir / ("optuna_best_params_" + study_name + ".json");
		fs::path curve_csv = out_dir / ("optuna_curve_best_" + study_name + ".csv");

		write_history_csv(db, study_id, hist_csv);
		if (bt.trial_id != -1) {
			auto params = get_trial_params(db, bt.trial_id);
			write_best_params_json(params, study_id, bt.number, bt.value, dir, best_json);
			write_intermediate_csv(db, bt.trial_id, curve_csv);
		}

		std::cout << BLUE << BOLD << "\n[Study] " << study_name << " (id=" << study_id << ")"
			<< " dir=" << (dir == 1 ? "maximize" : "minimize") << RESET << std::endl;
		if (bt.trial_id == -1) {
			std::cout << YELLOW << "  (sin trials completos)" << RESET << std::endl;
		}
		else {
			std::cout << GREEN << "  Mejor trial: #" << bt.number
				<< " | value=" << bt.value
				<< " | trial_id=" << bt.trial_id << RESET << std::endl;
			std::cout << "  Archivos: \n"
				<< "   - " << hist_csv.filename().string() << "\n"
				<< "   - " << best_json.filename().string() << "\n"
				<< "   - " << curve_csv.filename().string() << std::endl;
			any = true;
		}
	}

	sqlite3_close(db);
	return any;
}
