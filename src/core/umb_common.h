#include <assert.h>
#include <stdint.h>

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t  i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef bool b32;

typedef float  f32;
typedef double f64;

typedef char        byte;
typedef const char* str;

#define UMB_CLAMP(v, h, l)  ((v) > (h)) ? (h) : ((v) < (l)) ? l : v
#define UMB_CLAMP_BOT(v, l) ((v) < (l)) ? l : v
#define UMB_CLAMP_TOP(v, h) ((v) > (h)) ? h : v

#define UMB_KILOBYTES(n) n * 1024
#define UMB_MEGABYTES(n) UMB_KILOBYTES(n) * 1024
#define UMB_GIGAYTES(n)  UMB_MEGABYTES(n) * 1024

#define UMB_ASSERT(x) assert(x)

#define UMB_ARRAY_COUNT(arr, T) sizeof(arr) / sizeof(T)
