#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include "esp_heap_caps.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
#include "sdkconfig.h"

namespace {
constexpr size_t kInternalArenaCapacity = 180 * 1024;
constexpr size_t kPsramArenaCapacity = 384 * 1024;
constexpr int kBenchmarkRuns = 100;

void fail(const char* step) {
  printf("SB_ERROR,step=%s\n", step);
  fflush(stdout);
  while (true) vTaskDelay(pdMS_TO_TICKS(1000));
}

void print_heap(const char* stage) {
  printf("SB_HEAP,stage=%s,internal_free=%u,internal_min_free=%u,internal_largest=%u,psram_free=%u,psram_min_free=%u\n",
         stage,
         static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
         static_cast<unsigned>(heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
         static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
         static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)),
         static_cast<unsigned>(heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)));
}
}  // namespace

extern "C" void app_main(void) {
  vTaskDelay(pdMS_TO_TICKS(5000));
  printf("SB_META,model_sha256=%s,model_bytes=%u,runs=%d,cpu_mhz=%d,board=XIAO_ESP32S3,mode=precomputed_MFCC_no_peripherals\n",
         kModelSha256, static_cast<unsigned>(kModelDataLength), kBenchmarkRuns,
         CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
  print_heap("before_model");

  const tflite::Model* model = tflite::GetModel(kModelData);
  if (model->version() != TFLITE_SCHEMA_VERSION) fail("schema_version");
  static tflite::MicroMutableOpResolver<5> resolver;
  if (resolver.AddConv2D() != kTfLiteOk ||
      resolver.AddMaxPool2D() != kTfLiteOk ||
      resolver.AddMean() != kTfLiteOk ||
      resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddSoftmax() != kTfLiteOk) fail("resolver");

  size_t arena_capacity = kInternalArenaCapacity;
  const char* arena_location = "internal";
  uint8_t* arena = static_cast<uint8_t*>(heap_caps_malloc(arena_capacity, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  if (!arena) {
    arena_capacity = kPsramArenaCapacity;
    arena_location = "psram";
    arena = static_cast<uint8_t*>(heap_caps_malloc(arena_capacity, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  }
  if (!arena) fail("arena_alloc");
  print_heap("after_arena_alloc");

  alignas(tflite::MicroInterpreter) static unsigned char interpreter_storage[sizeof(tflite::MicroInterpreter)];
  auto* interpreter = new (interpreter_storage) tflite::MicroInterpreter(model, resolver, arena, arena_capacity);
  const int64_t allocation_started = esp_timer_get_time();
  if (interpreter->AllocateTensors() != kTfLiteOk) fail("allocate_tensors");
  const int64_t allocation_us = esp_timer_get_time() - allocation_started;
  TfLiteTensor* input = interpreter->input(0);
  TfLiteTensor* output = interpreter->output(0);
  if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8 ||
      input->bytes != 40 * 173 || output->bytes != 10) fail("tensor_shape_or_type");
  printf("SB_ALLOC,arena_location=%s,arena_capacity=%u,arena_used=%u,allocate_us=%lld,input_bytes=%u,output_bytes=%u,input_scale=%.9g,input_zero=%d,output_scale=%.9g,output_zero=%d\n",
         arena_location, static_cast<unsigned>(arena_capacity),
         static_cast<unsigned>(interpreter->arena_used_bytes()),
         static_cast<long long>(allocation_us),
         static_cast<unsigned>(input->bytes), static_cast<unsigned>(output->bytes),
         static_cast<double>(input->params.scale), static_cast<int>(input->params.zero_point),
         static_cast<double>(output->params.scale), static_cast<int>(output->params.zero_point));
  print_heap("after_allocate_tensors");

  // Golden vectors verify that the microcontroller really evaluates the
  // same INT8 model as the computer. Values come from TRAINING examples only.
  for (int sample = 0; sample < 3; ++sample) {
    memcpy(input->data.int8, kGoldenInputs[sample], input->bytes);
    if (interpreter->Invoke() != kTfLiteOk) fail("golden_invoke");
    int max_difference = 0;
    int predicted = 0;
    for (int cls = 0; cls < 10; ++cls) {
      const int diff = std::abs(static_cast<int>(output->data.int8[cls]) -
                                static_cast<int>(kGoldenOutputs[sample][cls]));
      max_difference = std::max(max_difference, diff);
      if (output->data.int8[cls] > output->data.int8[predicted]) predicted = cls;
    }
    printf("SB_GOLDEN,index=%d,file=%s,predicted=%d,max_int8_difference=%d,pass=%d\n",
           sample, kGoldenNames[sample], predicted, max_difference, max_difference <= 2);
  }

  // Warm-up runs are excluded from the timed sample.
  for (int i = 0; i < 5; ++i) {
    memcpy(input->data.int8, kGoldenInputs[i % 3], input->bytes);
    if (interpreter->Invoke() != kTfLiteOk) fail("warmup_invoke");
  }
  for (int i = 0; i < kBenchmarkRuns; ++i) {
    memcpy(input->data.int8, kGoldenInputs[i % 3], input->bytes);
    const int64_t start = esp_timer_get_time();
    if (interpreter->Invoke() != kTfLiteOk) fail("timed_invoke");
    const int64_t elapsed = esp_timer_get_time() - start;
    printf("SB_RUN,index=%d,invoke_us=%lld,internal_free=%u,psram_free=%u\n",
           i, static_cast<long long>(elapsed),
           static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
           static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)));
  }
  print_heap("after_runs");

  // This checks memory coexistence and timing of a synthetic 4-second PCM
  // producer. It is NOT an I2S recording or MFCC implementation.
  constexpr size_t kAudioSamples = 4 * 22050;
  auto* pcm = static_cast<int16_t*>(heap_caps_malloc(kAudioSamples * sizeof(int16_t),
                                                     MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
  if (!pcm) fail("sim_pcm_alloc");
  print_heap("with_sim_pcm");
  uint32_t pseudo_audio = 1;
  size_t written = 0;
  const int64_t simulated_capture_start = esp_timer_get_time();
  for (int chunk = 0; chunk < 400; ++chunk) {
    const size_t next = std::min(kAudioSamples, written + 221);
    for (size_t i = written; i < next; ++i) {
      pseudo_audio = pseudo_audio * 1664525u + 1013904223u;
      pcm[i] = static_cast<int16_t>(pseudo_audio >> 16);
    }
    written = next;
    vTaskDelay(pdMS_TO_TICKS(10));
  }
  const int64_t capture_us = esp_timer_get_time() - simulated_capture_start;
  const int64_t feature_load_start = esp_timer_get_time();
  memcpy(input->data.int8, kGoldenInputs[0], input->bytes);
  const int64_t feature_load_us = esp_timer_get_time() - feature_load_start;
  const int64_t sim_invoke_start = esp_timer_get_time();
  if (interpreter->Invoke() != kTfLiteOk) fail("sim_invoke");
  const int64_t sim_invoke_us = esp_timer_get_time() - sim_invoke_start;
  const int64_t post_start = esp_timer_get_time();
  int sums[10] = {};
  for (int repeat = 0; repeat < 5; ++repeat) {
    for (int cls = 0; cls < 10; ++cls) sums[cls] += output->data.int8[cls];
  }
  int chosen = 0;
  for (int cls = 1; cls < 10; ++cls) if (sums[cls] > sums[chosen]) chosen = cls;
  const int64_t post_us = esp_timer_get_time() - post_start;
  const int64_t ui_start = esp_timer_get_time();
  vTaskDelay(pdMS_TO_TICKS(20));  // Deliberate substitute for absent peripherals.
  const int64_t ui_wait_us = esp_timer_get_time() - ui_start;
  printf("SB_SIM,pcm_bytes=%u,capture_wait_us=%lld,feature_load_us=%lld,mfcc_measured=0,invoke_us=%lld,postprocess_us=%lld,ui_wait_us=%lld,predicted=%d\n",
         static_cast<unsigned>(kAudioSamples * sizeof(int16_t)),
         static_cast<long long>(capture_us), static_cast<long long>(feature_load_us),
         static_cast<long long>(sim_invoke_us), static_cast<long long>(post_us),
         static_cast<long long>(ui_wait_us), chosen);
  heap_caps_free(pcm);
  printf("SB_DONE\n");
  fflush(stdout);
  vTaskDelay(pdMS_TO_TICKS(10000));
  esp_restart();
}
