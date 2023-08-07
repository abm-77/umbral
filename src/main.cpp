#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <stdarg.h>
#include <stdio.h>
#include <umbral.h>

void log_proc(umb_log_message_type log_type, void* user_data, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  printf(fmt, args);
  va_end(args);
}

void start(umb_app* app) {
}

void update(umb_app* app) {
}

void shutdown(umb_app* app) {
}

int main(void) {
  umb_init_info init_info {.log_proc = log_proc};
  umb_init(&init_info);

  umb_app app;
  umb_app_init(&app, "[umbral]", 640, 480, start, update, shutdown);

  umb_app_run(&app);

  umb_shutdown();

  return 0;
}
