#include <stdio.h>
#include <stdlib.h>
#include <umbral.h>

umb_log_info umbi_log_info = {
    .verbosity = UMB_LOG_VERBOSITY_DEFAULT,
    .user_data = NULL,
    .log_proc  = NULL,
};

void* umb_default_alloc(size_t obj_size, size_t n_objs) {
  return malloc(obj_size * n_objs);
}

void umb_default_free(void* ptr, size_t s, size_t n) {
  free(ptr);
}

const umb_allocation_callbacks UMB_DEFAULT_ALLOC_CB = {umb_default_alloc, umb_default_free};

const umb_allocation_callbacks* UMB_ALLOC_CB = &UMB_DEFAULT_ALLOC_CB;

void ubmi_set_allocation_callbacks(const umb_allocation_callbacks* callbacks) {
  if (callbacks == NULL) {
    UMB_ALLOC_CB = &UMB_DEFAULT_ALLOC_CB;
  } else {
    UMB_ALLOC_CB = callbacks;
  }
}
