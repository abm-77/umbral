#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <gfx/umb_gfx.h>
#include <stdarg.h>
#include <stdio.h>
#include <umbral.h>

void log_proc(umb_log_message_type log_type, void* user_data, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  printf(fmt, args);
  printf("\n");
  va_end(args);
}

static umb_mesh          triangle_mesh;
static umb_render_object triangle;

static umb_mesh          monkey_mesh;
static umb_render_object monkey;

void start(umb_app* app) {
  UMBI_LOG_INFO("Starting [umbral]...");

  triangle_mesh = umb_mesh_create(3);

  umb_mesh_push_vertex(
      triangle_mesh,
      {
          .position = {1.0f, 1.0f, 0.5f},
          .color    = {0.f, 1.f, 0.0f},
      });
  umb_mesh_push_vertex(
      triangle_mesh,
      {
          .position = {-1.0f, 1.0f, 0.5f},
          .color    = {0.f, 1.f, 0.0f},
      });
  umb_mesh_push_vertex(
      triangle_mesh,
      {
          .position = {0.0f, -1.0f, 0.5f},
          .color    = {0.f, 1.f, 0.0f},
      });

  umb_gfx_register_mesh("triangle_mesh", triangle_mesh);

  triangle.mesh     = umb_gfx_get_mesh("triangle_mesh");
  triangle.material = umb_gfx_get_material("default");

  umb_mesh monk_mesh = umb_mesh_load_from_obj("res/models/monkey_smooth.obj");
  umb_gfx_register_mesh("monkey_mesh", monk_mesh);

  monkey.mesh      = umb_gfx_get_mesh("monkey_mesh");
  monkey.material  = umb_gfx_get_material("default");
  monkey.transform = glm::translate(glm::mat4(1), glm::vec3(0.0f, 5.0f, 0.0f));
}

void update(umb_app* app) {
}

void shutdown(umb_app* app) {
  UMBI_LOG_INFO("Shutting down [umbral]...");
}

int main(void) {
  umb_init_info init_info {.log_proc = log_proc};
  umb_init(&init_info);

  umb_app app;
  umb_app_init(&app, "[umbral]", 640, 480, start, update, shutdown);

  // umb_gfx_draw_object(&triangle);
  umb_gfx_draw_object(&monkey);

  umb_app_run(&app);

  umb_shutdown();

  return 0;
}
