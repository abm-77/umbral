#pragma once

#include <core/umb_common.h>
#include <stdlib.h>

#pragma region errors
enum umb_error {
  UMB_ERROR_OK = 0,
  UMB_ERROR_OUT_OF_MEM,
  UMB_ERROR_OBJECT_CREATION_FAILED,
  UMB_ERROR_OUT_OF_BOUNDS,
  UMB_ERROR_INVALID_FORMAT,
  UMB_ERROR_INVALID_SIZE,
  UMB_ERROR_INVALID_ENUM,
  UMB_ERROR_INVALID_OPERATION,
};
#pragma endregion

#pragma region logging
enum umb_log_message_type {
  UMB_LOG_INFO,
  UMB_LOG_WARNING,
  UMB_LOG_ERROR,
};

enum umb_log_verbosity {
  UMB_LOG_VERBOSITY_DEFAULT,
  UMB_LOG_VERBOSITY_DETAILED,
};

typedef void (*umb_log_proc)(umb_log_message_type, void*, const char*, ...);

struct umb_log_info {
  umb_log_verbosity verbosity;
  void*             user_data;
  umb_log_proc      log_proc;
};

extern umb_log_info umbi_log_info;

#define UMBI_LOG_MSG(level, fmt, ...)                                           \
  if (umbi_log_info.log_proc) {                                                 \
    umbi_log_info.log_proc(level, umbi_log_info.user_data, fmt, ##__VA_ARGS__); \
  }
#define UMBI_LOG_INFO(fmt, ...)  UMBI_LOG_MSG(UMB_LOG_INFO, fmt, ##__VA_ARGS__)
#define UMBI_LOG_WARN(fmt, ...)  UMBI_LOG_MSG(UMB_LOG_WARNING, fmt, ##__VA_ARGS__)
#define UMBI_LOG_ERROR(fmt, ...) UMBI_LOG_MSG(UMB_LOG_ERROR, fmt, ##__VA_ARGS__)

#pragma endregion

#pragma region memory
struct umb_allocation_callbacks {
  void* (*allocate)(size_t obj_size, size_t n_objs);
  void (*free)(void* ptr, size_t obj_size, size_t n_objs);
};

struct umb_arena_t {
  byte* data;
  u64   cap;
  u64   alloc_pos;
};
typedef struct umb_arena_t* umb_arena;

#define umb_arena_push_array(a, T, c) (T*)umb_arena_alloc(a, sizeof(T) * c)
#define umb_arena_push(a, T)          umb_arena_push_array(a, T, 1)

umb_arena_t umb_arena_create(u64 cap);
void*       umb_arena_alloc(umb_arena arena, u64 alloc_size);
void        umb_arena_dealloc(umb_arena arena, u64 dealloc_size);
void        umb_arena_dealloc_to(umb_arena arena, u64 pos);
void        umb_arena_clear(umb_arena arena);
void        umb_arena_release(umb_arena arena);
struct umb_temp_arena {
  umb_arena arena;
  u64       start_pos;
};
umb_temp_arena umb_temp_arena_create(umb_arena arena);
void           umb_temp_arena_end(umb_temp_arena* tmp);

class umb_scope_arena {
  public:
  umb_scope_arena(umb_arena arena) {
    tmp = umb_temp_arena_create(arena);
  }

  ~umb_scope_arena() {
    umb_temp_arena_end(&tmp);
  }

  private:
  umb_temp_arena tmp;
};

#define UMB_SLICE_DEF(T) \
  typedef struct {       \
    T*        data;      \
    const u32 len;       \
  } umb_slice_##T

#define UMB_SLICE(T, pdata, l) \
  umb_slice_##T {              \
    .data = pdata, .len = l,   \
  }

#define UMB_ARRAY_DEF(T) \
  typedef struct {       \
    T*  data;            \
    u32 len;             \
    u32 cap;             \
  } umb_array_##T

#define UMB_ARRAY_CREATE(T, arena, capacity)                                    \
  umb_array_##T {                                                               \
    .data = umb_arena_push_array(arena, T, capacity), .len = 0, .cap = capacity \
  }

#define UMB_PTR_ARRAY_DEF(T) \
  typedef struct {           \
    T** data;                \
    u32 len;                 \
    u32 cap;                 \
  } umb_ptr_array_##T
#define UMB_PTR_ARRAY_CREATE(T, arena, capacity)                                 \
  umb_ptr_array_##T {                                                            \
    .data = umb_arena_push_array(arena, T*, capacity), .len = 0, .cap = capacity \
  }

#define UMB_ARRAY_PUSH(arr, val) \
  UMB_ASSERT(arr.len < arr.cap); \
  arr.data[arr.len++] = val;

#define UMB_CONTAINER_DEF(T) \
  UMB_ARRAY_DEF(T);          \
  UMB_PTR_ARRAY_DEF(T);      \
  UMB_SLICE_DEF(T);

UMB_CONTAINER_DEF(str);
UMB_CONTAINER_DEF(byte);

#pragma endregion

#pragma region app
struct umb_init_info {
  umb_log_proc log_proc;
};

struct umb_window {
  str   title;
  u32   width;
  u32   height;
  void* raw_handle;
};

umb_error umb_window_init(umb_window* window, str title, u32 width, u32 height);
void      umb_window_destroy(umb_window* window);

struct umb_app;
typedef void (*umb_app_start_proc)(umb_app* app);
typedef void (*umb_app_update_proc)(umb_app* app);
typedef void (*umb_app_shutdown_proc)(umb_app* app);

struct umb_app {
  umb_window window;
  b32        running;

  umb_app_start_proc    start_proc;
  umb_app_update_proc   update_proc;
  umb_app_shutdown_proc shutdown_proc;
};

umb_error umb_init(umb_init_info* info);
umb_error umb_app_init(
    umb_app*              app,
    str                   title,
    u32                   width,
    u32                   height,
    umb_app_start_proc    start_proc,
    umb_app_update_proc   update_proc,
    umb_app_shutdown_proc shutdown_proc);
void umb_app_run(umb_app* app);
void umb_shutdown();

#pragma endregion

#pragma region io
umb_array_byte umb_read_file_binary(umb_arena arena, str filename);
#pragma endregion
