#include "fio.h"
#include "http/http.h"

static void handle_http_request(http_s *request) {
  const char body[] = "Hello from facil.io + Gleam!";
  http_send_body(request, (void *)body, sizeof(body) - 1);
}

void facil_io_demo_run(void) {
  if (http_listen("3000", NULL,
                  .on_request = handle_http_request,
                  .log = 1,
                  .public_folder = NULL) == -1) {
    perror("facil.io failed to listen on port 3000");
    return;
  }

  fio_start(.threads = 1, .workers = 1);
}
