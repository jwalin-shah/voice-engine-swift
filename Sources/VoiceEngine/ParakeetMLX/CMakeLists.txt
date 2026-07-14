cmake_minimum_required(VERSION 3.20)
project(ParakeetMLX LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find MLX
set(MLX_DIR "/tmp/parakeet-venv/lib/python3.12/site-packages/mlx/share/cmake/MLX")
find_package(MLX REQUIRED)

# Library target
add_library(parakeet_mlx STATIC
    weight_loader.cpp
    audio_preprocess.cpp
    conformer_encoder.cpp
    tdt_decoder.cpp
    tokenizer.cpp
    parakeet_engine.cpp
)

target_include_directories(parakeet_mlx PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${MLX_INCLUDE_DIRS}
)

target_link_libraries(parakeet_mlx PUBLIC
    mlx
)

target_compile_options(parakeet_mlx PRIVATE
    ${MLX_CXX_FLAGS}
)

# CLI test executable
add_executable(parakeet_cli parakeet_cli.cpp)
target_link_libraries(parakeet_cli PRIVATE parakeet_mlx)
target_compile_options(parakeet_cli PRIVATE ${MLX_CXX_FLAGS})

# Decoder test
add_executable(decoder_test decoder_test.cpp)
target_link_libraries(decoder_test PRIVATE parakeet_mlx)
target_compile_options(decoder_test PRIVATE ${MLX_CXX_FLAGS})

# Activation bisector
add_executable(bisector bisector.cpp)
target_link_libraries(bisector PRIVATE parakeet_mlx)
target_compile_options(bisector PRIVATE ${MLX_CXX_FLAGS})
