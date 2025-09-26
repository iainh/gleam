// Temporary shim for Cranelift stdlib inspect
void inspect(void* value) {
  (void)value;
}

#include "fio.h"
#include "http/http.h"
#include "gleam_ffi.h"

#include <stdint.h>
#include <stdio.h>

extern uint64_t gleam$facil_io_handle_request__1(uint64_t request_value);

void facil_io_send_body(uint64_t request_value, uint64_t body_value) {
  GleamFfiResourcePtr request_ptr = gleam_ffi_resource_to_ptr(request_value);
  if (request_ptr.status != GLEAM_FFI_STATUS_OK || request_ptr.ptr == NULL) {
    return;
  }

  GleamFfiBytes body = gleam_ffi_string_to_utf8(body_value);
  if (body.status != GLEAM_FFI_STATUS_OK) {
    gleam_ffi_bytes_free(body);
    return;
  }

  http_s *request = (http_s *)request_ptr.ptr;
  if (body.len == 0 || body.ptr == NULL) {
    http_send_body(request, "", 0);
  } else {
    http_send_body(request, body.ptr, body.len);
  }
  gleam_ffi_bytes_free(body);
}

static void handle_http_request(http_s *request) {
  GleamFfiValue resource = gleam_ffi_resource_from_ptr((void *)request);
  if (resource.status != GLEAM_FFI_STATUS_OK) {
    return;
  }

  (void)gleam$facil_io_handle_request__1(resource.value);
}

void facil_io_run(void) {
  if (http_listen("3000", NULL,
                  .on_request = handle_http_request,
                  .log = 1,
                  .public_folder = NULL) == -1) {
    perror("facil.io failed to listen on port 3000");
    return;
  }

  fio_start(.threads = 1, .workers = 1);
}
