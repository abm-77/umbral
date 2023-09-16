#version 460

layout(location = 0) in vec3 in_col;
layout(location = 1) in vec2 in_texcoord;

layout(location = 0) out vec4 out_col;

layout(set = 0, binding = 1) uniform SceneData {
  vec4 fog_color;
  vec4 fog_distances;
  vec4 ambient_color;
  vec4 sunlight_direction;
  vec4 sunlight_color;
} scene_data;


void main() {
  out_col = vec4(in_texcoord.x, in_texcoord.y, 0.5f, 1.0f);
}
