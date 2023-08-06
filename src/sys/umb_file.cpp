#include <stdio.h>
#include <umbral.h>

umb_array_byte umb_read_file_binary(umb_arena arena, str filename) {
  FILE* fp = fopen(filename, "rb");
  UMB_ASSERT(fp);

  fseek(fp, 0, SEEK_END);
  u32 file_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  umb_array_byte file_data = UMB_ARRAY_CREATE(byte, arena, file_size);
  fread(file_data.data, file_size, 1, fp);
  file_data.len = file_size;

  return file_data;
}
