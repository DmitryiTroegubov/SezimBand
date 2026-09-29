#pragma once

#include <cstddef>
#include <cstdint>

extern const unsigned char kModelData[];
extern const size_t kModelDataLength;
extern const int8_t kGoldenInputs[3][40 * 173];
extern const int8_t kGoldenOutputs[3][10];
extern const char* const kGoldenNames[3];
extern const char kModelSha256[];

