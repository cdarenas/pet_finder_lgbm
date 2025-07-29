cmake_minimum_required(VERSION 3.22)
project(PetFinderLGBM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Agrega vcpkg toolchain solo si no integraste globalmente
#set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
# Incluir directorios
include_directories(include)

# Agregar todos los archivos fuente
file(GLOB SOURCES
    src/*.cpp
)

# Dependencias
#find_package(SQLite3 REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)

add_executable(PetFinderLGBM ${SOURCES}
)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# Copiar lightgbm.exe al binario final
add_custom_command(TARGET PetFinderLGBM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_SOURCE_DIR}/lightgbm.exe
    $<TARGET_FILE_DIR:PetFinderLGBM>
)

# Copiar carpeta scripts al directorio de salida
add_custom_command(TARGET PetFinderLGBM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/scripts
    $<TARGET_FILE_DIR:PetFinderLGBM>/scripts
)

# Copiar carpeta folds
add_custom_command(TARGET PetFinderLGBM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/folds
        $<TARGET_FILE_DIR:PetFinderLGBM>/folds
)

# fast-cpp-csv-parser es header-only
find_path(FAST_CPP_CSV_PARSER_INCLUDE_DIRS "fast-cpp-csv-parser/csv.h")
target_include_directories(PetFinderLGBM PRIVATE ${FAST_CPP_CSV_PARSER_INCLUDE_DIRS})

# Linkea librerías
target_link_libraries(PetFinderLGBM
    PRIVATE
    unofficial::sqlite3::sqlite3
)

# Para compilar con ruta correcta desde Visual Studio
if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()
