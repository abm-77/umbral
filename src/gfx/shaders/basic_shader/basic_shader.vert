#version 460

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_norm;
layout(location = 2) in vec3 in_col;
layout(location = 3) in vec2 in_texcoord;

layout(location = 0) out vec3 out_col;
layout(location = 1) out vec2 out_tecoord;

layout(set = 0, binding = 0) uniform CameraBuffer {
  mat4 view;
  mat4 proj;
  mat4 viewproj;
} camera_data;

struct ObjectData {
  mat4 model;
};

layout(std140, set = 1, binding = 0) readonly buffer ObjectBuffer{
  ObjectData objects[];
} object_buffer;

layout( push_constant )  uniform PushConstants {
  vec4 data;
  mat4 render_matrix;
} push_constants;

void main() {
  mat4 model_matrix = object_buffer.objects[gl_BaseInstance].model;
  mat4 transform_matrix = (camera_data.viewproj * model_matrix);
  gl_Position =  transform_matrix * vec4(in_pos, 1.0);
  out_col = in_col;
  out_tecoord = in_texcoord;
}

