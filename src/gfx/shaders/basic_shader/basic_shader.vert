#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_norm;
layout(location = 2) in vec3 in_col;

layout(location = 0) out vec3 out_col;

layout( push_constant )  uniform constants {
  vec4 data;
  mat4 render_matrix;
} push_constants;

void main() {
  gl_Position = push_constants.render_matrix * vec4(in_pos, 1.0);
  out_col = in_col;
}

