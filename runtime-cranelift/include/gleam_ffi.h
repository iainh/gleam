#ifndef GLEAM_FFI_H
#define GLEAM_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  GLEAM_FFI_STATUS_OK = 0,
  GLEAM_FFI_STATUS_NOT_A_STRING = 1,
  GLEAM_FFI_STATUS_MISSING_DATA = 2,
  GLEAM_FFI_STATUS_INVALID_UTF8 = 3,
  GLEAM_FFI_STATUS_NOT_AN_INT = 4,
  GLEAM_FFI_STATUS_NEGATIVE_INT = 5,
  GLEAM_FFI_STATUS_INVALID_ARGUMENT = 6,
  GLEAM_FFI_STATUS_NOT_A_RESOURCE = 7,
  GLEAM_FFI_STATUS_NULL_POINTER = 8,
  GLEAM_FFI_STATUS_NOT_A_LIST = 9
} GleamFfiStatus;

typedef struct {
  GleamFfiStatus status;
  uint8_t *ptr;
  size_t len;
} GleamFfiBytes;

typedef struct {
  GleamFfiStatus status;
  uint64_t value;
} GleamFfiUint;

typedef struct {
  GleamFfiStatus status;
  uint64_t value;
} GleamFfiValue;

typedef struct {
  GleamFfiStatus status;
  void *ptr;
} GleamFfiResourcePtr;

typedef struct {
  GleamFfiStatus status;
  uint64_t *ptr;
  size_t len;
} GleamFfiValues;

GleamFfiBytes gleam_ffi_string_to_utf8(uint64_t value);
void gleam_ffi_bytes_free(GleamFfiBytes bytes);
GleamFfiValue gleam_ffi_string_from_utf8(const uint8_t *ptr, size_t len);
uint64_t gleam_ffi_encode_uint(uint64_t value);
GleamFfiUint gleam_ffi_decode_uint(uint64_t value);
GleamFfiValue gleam_ffi_resource_from_ptr(void *ptr);
GleamFfiResourcePtr gleam_ffi_resource_to_ptr(uint64_t value);
GleamFfiValue gleam_ffi_list_from_array(const uint64_t *ptr, size_t len);
GleamFfiValues gleam_ffi_list_to_array(uint64_t list_value);
void gleam_ffi_values_free(GleamFfiValues values);

#ifdef __cplusplus
}
#endif

#endif // GLEAM_FFI_H
