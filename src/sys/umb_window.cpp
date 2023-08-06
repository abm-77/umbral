#include <SDL2/SDL.h>
#include <core/umb_common.h>
#include <umbral.h>

umb_error umb_window_init(umb_window* pwindow, str title, u32 width, u32 height) {
  SDL_Window* window = SDL_CreateWindow(
      title,
      SDL_WINDOWPOS_UNDEFINED,
      SDL_WINDOWPOS_UNDEFINED,
      width,
      height,
      SDL_WINDOW_SHOWN | SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);

  if (!window) {
    UMBI_LOG_ERROR("Failed to create SDL Window: %s", SDL_GetError());
    return UMB_ERROR_OBJECT_CREATION_FAILED;
  }

  pwindow->title      = title;
  pwindow->width      = width;
  pwindow->height     = height;
  pwindow->raw_handle = window;

  return UMB_ERROR_OK;
}

void umb_window_destroy(umb_window* window) {
  SDL_DestroyWindow((SDL_Window*)window->raw_handle);
}
