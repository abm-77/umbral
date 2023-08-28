#include <SDL2/SDL.h>
#include <core/umb_common.h>
#include <gfx/umb_gfx.h>

umb_error umb_init(umb_init_info* init_info) {
  umb_error err = UMB_ERROR_OK;
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    UMBI_LOG_ERROR("Could initialize SDL!");
    err = UMB_ERROR_OBJECT_CREATION_FAILED;
  }

  if (init_info) umbi_log_info.log_proc = init_info->log_proc;

  return err;
}

umb_error umb_app_init(
    umb_app*              app,
    str                   title,
    u32                   width,
    u32                   height,
    umb_app_start_proc    start_proc,
    umb_app_update_proc   update_proc,
    umb_app_shutdown_proc shutdown_proc) {
  umb_error err = umb_window_init(&app->window, title, width, height);
  if (err != UMB_ERROR_OK) {
    UMBI_LOG_ERROR("Could not create Umbral App!");
    return err;
  }

  app->running       = true;
  app->start_proc    = start_proc;
  app->update_proc   = update_proc;
  app->shutdown_proc = shutdown_proc;

  umb_gfx_init(&app->window);

  return UMB_ERROR_OK;
}

void umb_app_run(umb_app* app) {
  if (app->start_proc) app->start_proc(app);

  while (app->running) {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
      switch (e.type) {
      case SDL_QUIT: {
        app->running = false;
      } break;

      case SDL_WINDOWEVENT: {
        switch (e.window.event) {
        case SDL_WINDOWEVENT_RESIZED:
        case SDL_WINDOWEVENT_SIZE_CHANGED: {
          umb_gfx_framebuffer_resized();
        } break;
        }
      } break;
      }

      if (app->update_proc) app->update_proc(app);
    }

    umb_gfx_draw_frame();
  }

  if (app->shutdown_proc) app->shutdown_proc(app);
  umb_window_destroy(&app->window);
}

void umb_shutdown() {
  umb_gfx_shutdown();
  SDL_Quit();
}
