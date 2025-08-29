#include "database.hpp"
#include <iostream>
#include <fstream>
#include <ctime>
#include <sqlite3.h>
#include <filesystem>

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

// Insertar resultados en la base de datos SQLite
int insert_result_sqlite(double acc, double f1, double kappa, const string& model_path, const string& conf_path, const string& config_str) {
	sqlite3* db;
	char* errMsg = nullptr;
	int rc = sqlite3_open("resultados.db", &db);
	if (rc) {
		cerr << "No se puede abrir la base de datos: " << sqlite3_errmsg(db) << endl;
		return -1;
	}

	const char* create_sql = "CREATE TABLE IF NOT EXISTS resultados ("
		"id INTEGER PRIMARY KEY AUTOINCREMENT, "
		"fecha TEXT, "
		"accuracy REAL, "
		"f1_macro REAL, "
		"kappa REAL, "
		"modelo TEXT, "
		"config TEXT, "
		"config_text TEXT);";
	sqlite3_exec(db, create_sql, 0, 0, &errMsg);

	time_t now = time(0);
	string fecha = string(ctime(&now));
	fecha.pop_back(); // quitar salto de línea

	string sql = "INSERT INTO resultados (fecha, accuracy, f1_macro, kappa, modelo, config, config_text) VALUES (?, ?, ?, ?, ?, ?, ?);";
	sqlite3_stmt* stmt;
	sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr);
	sqlite3_bind_text(stmt, 1, fecha.c_str(), -1, SQLITE_STATIC);
	sqlite3_bind_double(stmt, 2, acc);
	sqlite3_bind_double(stmt, 3, f1);
	sqlite3_bind_double(stmt, 4, kappa);
	sqlite3_bind_text(stmt, 5, model_path.c_str(), -1, SQLITE_STATIC);
	sqlite3_bind_text(stmt, 6, conf_path.c_str(), -1, SQLITE_STATIC);
	sqlite3_bind_text(stmt, 7, config_str.c_str(), -1, SQLITE_STATIC);

	int id_resultado = -1;
	rc = sqlite3_step(stmt);
	if (rc == SQLITE_DONE) {
		id_resultado = static_cast<int>(sqlite3_last_insert_rowid(db));
	}
	else {
		cerr << "Error al insertar resultado: " << sqlite3_errmsg(db) << endl;
	}
	sqlite3_finalize(stmt);
	sqlite3_close(db);
	return id_resultado;
}

// Insertar predicciones en la base de datos SQLite
void insert_predictions_sqlite(const vector<int>& y_true, const vector<int>& y_pred, int result_id) {
	sqlite3* db;
	sqlite3_open("resultados.db", &db);

	const char* create_sql = "CREATE TABLE IF NOT EXISTS predicciones ("
		"id INTEGER PRIMARY KEY AUTOINCREMENT, "
		"id_resultado INTEGER, "
		"indice INTEGER, "
		"y_true INTEGER, "
		"y_pred INTEGER);";
	sqlite3_exec(db, create_sql, 0, 0, nullptr);

	const char* insert_sql = "INSERT INTO predicciones (id_resultado, indice, y_true, y_pred) VALUES (?, ?, ?, ?);";
	sqlite3_stmt* stmt;
	sqlite3_prepare_v2(db, insert_sql, -1, &stmt, nullptr);

	for (size_t i = 0; i < y_true.size(); ++i) {
		sqlite3_bind_int(stmt, 1, result_id);
		sqlite3_bind_int(stmt, 2, static_cast<int>(i));
		sqlite3_bind_int(stmt, 3, y_true[i]);
		sqlite3_bind_int(stmt, 4, y_pred[i]);

		if (sqlite3_step(stmt) != SQLITE_DONE) {
			cerr << "Error insertando prediccion: " << sqlite3_errmsg(db) << endl;
		}
		sqlite3_reset(stmt);
	}

	sqlite3_finalize(stmt);
	sqlite3_close(db);
}

// Guardar el mejor modelo basado en F1 macro
void save_best_model(const std::string& db_path) {
	sqlite3* db;
	sqlite3_open("resultados.db", &db);

	const char* create_best_sql = "CREATE TABLE IF NOT EXISTS mejor_modelo ("
		"id INTEGER PRIMARY KEY, modelo TEXT, id_resultado INTEGER, es_final INTEGER DEFAULT 0, configuracion TEXT);";
	sqlite3_exec(db, create_best_sql, nullptr, nullptr, nullptr);

	const char* best_sql = "SELECT id, modelo FROM resultados ORDER BY f1_macro DESC LIMIT 1;";
	sqlite3_stmt* stmt;
	if (sqlite3_prepare_v2(db, best_sql, -1, &stmt, nullptr) == SQLITE_OK && sqlite3_step(stmt) == SQLITE_ROW) {
		int best_id = sqlite3_column_int(stmt, 0);
		string best_model = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
		fs::copy_file(best_model, "best_model.txt", fs::copy_options::overwrite_existing);
		sqlite3_finalize(stmt);

		const char* insert_best_sql = "INSERT OR REPLACE INTO mejor_modelo (id, modelo, id_resultado) VALUES (1, ?, ?);";
		if (sqlite3_prepare_v2(db, insert_best_sql, -1, &stmt, nullptr) == SQLITE_OK) {
			sqlite3_bind_text(stmt, 1, best_model.c_str(), -1, SQLITE_STATIC);
			sqlite3_bind_int(stmt, 2, best_id);
			sqlite3_step(stmt);
			sqlite3_finalize(stmt);
		}
	}
	else {
		cerr << "Error al seleccionar mejor modelo." << endl;
	}
	sqlite3_close(db);
}

// Guardar el mejor modelo basado en Kappa
void save_best_model_by_kappa() {
	sqlite3* db;
	sqlite3_open("resultados.db", &db);

	const char* query = "SELECT id, modelo FROM resultados ORDER BY kappa DESC LIMIT 1;";
	sqlite3_stmt* stmt;
	if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) == SQLITE_OK && sqlite3_step(stmt) == SQLITE_ROW) {
		string modelo = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
		fs::copy_file(modelo, "mejor_modelo_kappa.txt", fs::copy_options::overwrite_existing);
	}
	else {
		cerr << "Error seleccionando mejor modelo por kappa." << endl;
	}
	sqlite3_finalize(stmt);
	sqlite3_close(db);
}

// Guardar el modelo final en la base de datos
void save_final_model(const std::string& db_path, const std::string& model_path, const std::string& config_path) {
	sqlite3* db;
	if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK) {
		std::cerr << "No se pudo abrir la base de datos." << std::endl;
		return;
	}

	const char* create_table_sql = R"(
	CREATE TABLE IF NOT EXISTS mejor_modelo (
		id INTEGER PRIMARY KEY,
		modelo TEXT NOT NULL,
		id_resultado INTEGER,
		es_final INTEGER DEFAULT 0,
		configuracion TEXT
	);)";
	sqlite3_exec(db, create_table_sql, nullptr, nullptr, nullptr);

	// Copiar el modelo final a "modelo_final.txt"
	std::filesystem::copy_file(model_path, "modelo_final.txt", std::filesystem::copy_options::overwrite_existing);

	// Insertar o reemplazar el modelo final
	const char* insert_sql = R"(
	INSERT OR REPLACE INTO mejor_modelo (id, modelo, id_resultado, es_final, configuracion)
	VALUES (1, ?, NULL, 1, ?);
	)";
	sqlite3_stmt* stmt;
	if (sqlite3_prepare_v2(db, insert_sql, -1, &stmt, nullptr) == SQLITE_OK) {
		sqlite3_bind_text(stmt, 1, model_path.c_str(), -1, SQLITE_STATIC);
		sqlite3_bind_text(stmt, 2, config_path.c_str(), -1, SQLITE_STATIC);
		sqlite3_step(stmt);
		sqlite3_finalize(stmt);
	}
	else {
		cerr << "Error al preparar el INSERT del modelo final: " << sqlite3_errmsg(db) << endl;
	}

	sqlite3_close(db);
	cout << MAGENTA << "📁 Modelo final guardado en: " << db_path << RESET << endl;
}

// Verificar si una tabla existe en la base de datos SQLite
bool sqlite_table_exists(const std::string& db_path, const std::string& table_name) {
	sqlite3* db = nullptr;
	if (sqlite3_open(db_path.c_str(), &db) != SQLITE_OK) {
		if (db) sqlite3_close(db);
		return false; // si no puedo abrir, asumimos que no existe (o DB ausente)
	}

	const char* sql =
		"SELECT name FROM sqlite_master "
		"WHERE type='table' AND name=?1 LIMIT 1;";

	sqlite3_stmt* stmt = nullptr;
	bool exists = false;

	if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
		sqlite3_bind_text(stmt, 1, table_name.c_str(), -1, SQLITE_TRANSIENT);
		int rc = sqlite3_step(stmt);
		if (rc == SQLITE_ROW) {
			exists = true;
		}
	}

	if (stmt) sqlite3_finalize(stmt);
	sqlite3_close(db);
	return exists;
}

