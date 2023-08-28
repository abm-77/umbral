#pragma once

#include <string.h>
#include <umbral.h>

#define DEFAULT_NUM_SLOTS 256

// djb2 hash
u64 hash_string(str s) {
  u64 result = 5381;
  for (i32 i = 0; i < strlen(s); i++) result = ((result << 5) + result) + s[i];
  return result;
}

struct umb_hash_node {
  str   key;
  byte* value;
  u64   hash;

  struct umb_hash_node* next;
};

struct umb_hash_table {
  umb_arena       arena;
  umb_hash_node** slots;
  u64             slot_count;
};

umb_hash_table umb_hash_table_create(umb_arena arena, u64 slot_count) {
  umb_hash_table ht = {
      .arena      = arena,
      .slots      = umb_arena_push_array(arena, umb_hash_node*, slot_count),
      .slot_count = slot_count,
  };
  return ht;
}

byte* umb_hash_table_get(umb_hash_table* ht, str key) {
  byte* result = NULL;

  u64 hash = hash_string(key);

  u64 slot_idx = hash % ht->slot_count;

  umb_hash_node* node = NULL;
  for (umb_hash_node* curr = ht->slots[slot_idx]; curr != NULL; curr = curr->next) {
    if (curr->hash == hash) {
      node = curr;
      break;
    }
  }

  if (node != NULL) result = node->value;

  return result;
}

void umb_hash_table_insert(umb_hash_table* ht, str key, byte* value) {
  u64 hash     = hash_string(key);
  u64 slot_idx = hash % ht->slot_count;

  umb_hash_node* existing_node = NULL;
  for (umb_hash_node* curr = ht->slots[slot_idx]; curr != NULL; curr = curr->next) {
    if (curr->hash == hash) {
      existing_node = curr;
      break;
    }
  }

  if (existing_node == NULL) {
    umb_hash_node* new_node = umb_arena_push(ht->arena, umb_hash_node);
    new_node->key           = key;

    new_node->value = value;

    new_node->hash      = hash;
    new_node->next      = ht->slots[slot_idx];
    ht->slots[slot_idx] = new_node;
  } else {
    existing_node->value = value;
  }
}

typedef struct umb_hash_set {
  umb_hash_table ht;
} umb_hash_set;

umb_hash_set umb_hash_set_create(umb_arena arena, u64 slot_count) {
  umb_hash_set hs = {.ht = umb_hash_table_create(arena, slot_count)};
  return hs;
}

void umb_hash_set_insert(umb_hash_set* hs, str key) {
  umb_hash_table_insert(&hs->ht, key, (byte*)1);
}

void umb_hash_set_remove(umb_hash_set* hs, str key) {
  umb_hash_table_insert(&hs->ht, key, (byte*)0);
}

b32 umb_hash_set_has(umb_hash_set* hs, str key) {
  byte* result = umb_hash_table_get(&hs->ht, key);
  return result != NULL && result == (byte*)1;
}
