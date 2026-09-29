idf_component_register(SRCS "main.cc" "model_data.cc"
                       INCLUDE_DIRS "."
                       REQUIRES esp_timer heap)
target_compile_options(${COMPONENT_LIB} PRIVATE -DESP_NN)

