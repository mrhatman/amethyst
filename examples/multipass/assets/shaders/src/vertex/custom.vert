#version 450


layout(location = 0) in vec2 pos;

layout(location = 0) out VertexData {
    vec2 pos;
} vertex;


void main() {

    vertex.pos = pos;

    vec4 position = vec4(pos, 0.0, 1.0);
    gl_Position = position;
}
