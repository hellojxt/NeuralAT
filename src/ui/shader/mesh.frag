#version 330 core

in vec2 TexCoord;
in vec3 Normal;
in vec3 Normal_static;
in vec3 FragPos;

out vec4 FragColor;

uniform sampler2D texture_diffuse;
uniform vec3 lightDir; // Direction of the light source
uniform vec3 lightColor; // Color of the light
uniform vec3 viewPos; // Position of the camera/viewer


void main()
{
    // Ambient Lighting
    float ambientStrength = 0.33; // You can adjust this value
    vec3 ambient = ambientStrength * lightColor;

    // Diffuse Lighting
    vec3 norm = normalize(Normal);
    float diff = max(dot(norm, normalize(lightDir)), 0.0);

    vec3 norm_static = normalize(Normal_static);
    float diff_static = max(dot(norm_static, normalize(lightDir)), 0.0);

    vec3 diffuse = (diff_static*0.5 + diff*0.5) * lightColor * 0.33;

    // Specular Lighting
    float specularStrength = 0.33; // You can adjust this value
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16); // 16 is the shininess factor, can be adjusted
    vec3 specular = specularStrength * spec * lightColor;

    // Combine the texture color with the lighting components
    vec4 texColor = texture(texture_diffuse, TexCoord);
    vec3 finalColor = (ambient + diffuse + specular) * texColor.rgb;

    FragColor = vec4(finalColor, texColor.a);
}
