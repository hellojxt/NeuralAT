#version 330 core

layout (location = 0) in vec3 aPos; // Position
layout (location = 1) in vec2 aTexCoord; // Texture coordinates
layout (location = 2) in vec3 aNormal; // Normal

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 displacement; // Displacement of the object

out vec2 TexCoord;
out vec3 Normal;
out vec3 Normal_static;
out vec3 FragPos;


void main()
{
    // Transform the vertex position to clip space
    gl_Position = projection * view * model * vec4(aPos + displacement, 1.0);

    // Pass the texture coordinates to the fragment shader
    TexCoord = aTexCoord;

    // Transform the normal vector to view space and pass it to the fragment shader
    Normal = mat3(transpose(inverse(view * model))) * aNormal;
    Normal_static = mat3(transpose(inverse(model))) * aNormal;

    // Transform the vertex position to view space and pass it to the fragment shader
    FragPos = vec3(model * vec4(aPos + displacement, 1.0));
}
