#pragma once

#include <SDL_vulkan.h>
#include <umbral.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct umb_mesh_vertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec3 color;
  glm::vec2 uv;
};
UMB_CONTAINER_DEF(umb_mesh_vertex);

struct umb_text_mesh_vertex {
  glm::vec3 position;
  glm::vec3 color;
};
UMB_CONTAINER_DEF(umb_text_mesh_vertex);

typedef struct umb_mesh_t*      umb_mesh;
typedef struct umb_text_mesh_t* umb_text_mesh;

struct umb_pipeline {
  VkPipeline       pipeline;
  VkPipelineLayout pipeline_layout;
};

struct umb_material {
  umb_pipeline pipeline;
};

struct umb_render_object {
  umb_mesh      mesh;
  umb_material* material;
  glm::mat4     transform;
};

typedef struct umb_image_t*   umb_image;
typedef struct umb_texture_t* umb_texture;

UMB_CONTAINER_DEF(umb_render_object);

void umb_gfx_init(umb_window* window);
void umb_gfx_draw_frame();
void umb_gfx_draw_object(umb_render_object* o);
void umb_gfx_shutdown();
void umb_gfx_framebuffer_resized();

void umb_gfx_register_mesh(str name, umb_mesh mesh);
void umb_gfx_register_material(str name, umb_material* mat);

b32 umb_gfx_load_image_from_file(str file, umb_image image);

// TODO(bryson): change to out pointer API
umb_mesh      umb_gfx_get_mesh(str name);
umb_material* umb_gfx_get_material(str name);

umb_mesh umb_mesh_create(u32 n_vertices);
umb_mesh umb_mesh_load_from_obj(str filename);
void     umb_mesh_push_vertex(umb_mesh mesh, umb_mesh_vertex vertex);
