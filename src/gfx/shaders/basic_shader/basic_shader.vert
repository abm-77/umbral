#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_norm;
layout(location = 2) in vec3 in_col;

layout(location = 0) out vec3 out_col;

vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

void main() {
  gl_Position = vec4(in_pos, 1.0);
  out_col = in_col;
}

