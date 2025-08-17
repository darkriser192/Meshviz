BASIC_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 position;
uniform mat4 projection;

void main() {
    gl_Position = projection * vec4(position, 1.0);
}
"""

BASIC_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;

void main() {
    FragColor = vec4(1.0, 0.0, 0.0, 1.0);  // Red color
}
"""

from OpenGL.GL import *

# shader_type can be GL_VERTEX_SHADER or GL_FRAGMENT_SHADER
def compile_single_shader(shader_type, source_code):
    # Create shader object (returns integer handle)
    shader = glCreateShader(shader_type)
    
    # Attach source code and compile
    glShaderSource(shader, source_code)
    glCompileShader(shader)

    # Check compile status
    success = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if success != GL_TRUE:
        info_log = glGetShaderInfoLog(shader).decode()
        shader_name = (
            "VERTEX" if shader_type == GL_VERTEX_SHADER 
            else "FRAGMENT" if shader_type == GL_FRAGMENT_SHADER 
            else "UNKNOWN"
        )
        raise RuntimeError(f"{shader_name} SHADER COMPILATION ERROR:\n{info_log}")
    
    return shader  # Return compiled shader object

def create_shader_program(vertex_source, fragment_source):
    vertex_shader = compile_single_shader(GL_VERTEX_SHADER, vertex_source)
    fragment_shader = compile_single_shader(GL_FRAGMENT_SHADER, fragment_source)

    # Create program and attach shaders
    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    # Check link status
    success = glGetProgramiv(program, GL_LINK_STATUS)
    if success != GL_TRUE:
        info_log = glGetProgramInfoLog(program).decode()
        raise RuntimeError(f"SHADER PROGRAM LINK ERROR:\n{info_log}")
    
    # Clean up (shaders are now linked into the program)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return program
