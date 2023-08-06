#include <assert.h>
#include <stdlib.h>
#include <umbral.h>
#include <memory.h>

umb_arena_t umb_arena_create(u64 cap) {
  return umb_arena_t {
      .data      = (byte*)malloc(cap),
      .cap       = cap,
      .alloc_pos = 0,
  };
}

void* umb_arena_alloc(umb_arena arena, u64 alloc_size) {
  assert(arena->alloc_pos + alloc_size < arena->cap);

  void* result = arena->data + arena->alloc_pos;
  memset(result, 0, alloc_size);
  arena->alloc_pos += alloc_size;
  return result;
}

void umb_arena_dealloc(umb_arena arena, u64 dealloc_size) {
  u64 clamped_size = UMB_CLAMP_TOP(dealloc_size, arena->alloc_pos);
  arena->alloc_pos -= clamped_size;
}

void umb_arena_dealloc_to(umb_arena arena, u64 pos) {
  u64 pos_diff = UMB_CLAMP_BOT(arena->alloc_pos - pos, 0);
  arena->alloc_pos -= pos_diff;
}

void umb_arena_clear(umb_arena arena) {
  arena->alloc_pos = 0;
}

void umb_arena_release(umb_arena arena) {
  arena->alloc_pos = 0;
  arena->cap       = 0;
  free(arena->data);
}

umb_temp_arena umb_temp_arena_create(umb_arena arena) {
  return umb_temp_arena {
      .arena     = arena,
      .start_pos = arena->alloc_pos,
  };
}

void umb_temp_arena_end(umb_temp_arena* tmp) {
  umb_arena_dealloc_to(tmp->arena, tmp->start_pos);
}
