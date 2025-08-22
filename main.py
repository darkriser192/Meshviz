import glfw
from OpenGL.GL import *  # pyright: ignore[reportWildcardImportFromLibrary]
import OpenGL.GLUT as GLUT # pyright: ignore[reportWildcardImportFromLibrary]
import numpy as np
from shaders.shaders import *
import shapes as shp
from utils.log_setup import setup_logger
from utils.sysinfo2 import get_system_info
from utils.sysinfo import get_system_limits
import sympy as smp 

# TODO optimize "import" statements across the codebase
# TODO Implement BVH
# TODO Implement some raytraicing

DEBUG = True  # Set to True for debugging output

logger = setup_logger() 

class ColorMode:
    INDIVIDUAL = "individual"      # Each mesh uses its solid_color
    OVERRIDE_SINGLE = "single"     # All meshes use one color  
    OVERRIDE_BY_TYPE = "by_type"   # Color by object type
    DYNAMIC_NORMALS = "normals"    # Per-facet based on vertex normals

# Configuration - change this and restart to switch modes
CURRENT_COLOR_MODE = ColorMode.INDIVIDUAL  # Change this to test different modes

COLOR_CONFIGS = {
    ColorMode.OVERRIDE_SINGLE: np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),  # Red
    ColorMode.OVERRIDE_BY_TYPE: {
        'tetrahedron': np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),  # Green
        'cube': np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),         # Blue
    }
}

def set_mesh_color(uniform_locations, render_mesh, color_mode, mesh_type=None):
    """Set mesh color based on current mode"""
    color_loc = uniform_locations['meshColor']
    
    if color_mode == ColorMode.INDIVIDUAL:
        glUniform4fv(color_loc, 1, render_mesh.solid_color)
    
    elif color_mode == ColorMode.OVERRIDE_SINGLE:
        override_color = COLOR_CONFIGS[ColorMode.OVERRIDE_SINGLE]
        glUniform4fv(color_loc, 1, override_color)
    
    elif color_mode == ColorMode.OVERRIDE_BY_TYPE:
        type_colors = COLOR_CONFIGS[ColorMode.OVERRIDE_BY_TYPE]
        color = type_colors.get(mesh_type, render_mesh.solid_color)
        glUniform4fv(color_loc, 1, color)
    
    elif color_mode == ColorMode.DYNAMIC_NORMALS:
        # For now, use a different color to show it's working
        normal_color = np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32)  # Yellow
        glUniform4fv(color_loc, 1, normal_color)
        # TODO: Implement actual normal-based coloring
    
    else:
        logger.error(f"Unknown color mode: {color_mode}")
        glUniform4fv(color_loc, 1, render_mesh.solid_color)

def framebuffer_size_callback(window, width, height):
    # Adjust the OpenGL viewport to the new window size
    glViewport(0, 0, width, height)
    glEnable(GL_DEPTH_TEST)
    glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT)  # Update clear call
    # Query window size (logical screen coordinates)
    window_width, window_height = glfw.get_window_size(window)
    # Log both sizes
    if DEBUG:
        logger.info(f"[Resize] Window size: {window_width}x{window_height}, Framebuffer size: {width}x{height}")

def create_orthographic_matrix(aspect_ratio = 1.0):
    left = -aspect_ratio
    right = aspect_ratio
    bottom = -1.0
    top = 1.0
    near = -1.0
    far = 1.0

    proj = np.array([
        [2/(right-left), 0, 0, -(right+left)/(right-left)],
        [0, 2/(top-bottom), 0, -(top+bottom)/(top-bottom)],
        [0, 0, -2/(far-near), -(far+near)/(far-near)],
        [0, 0, 0, 1]
    ], dtype=np.float64)

    return proj

def create_perspective_matrix(fov=45.0, aspect_ratio=1.0, near=0.1, far=100.0):
    f = 1.0 / np.tan(np.radians(fov) / 2.0)
    proj = np.array([
        [f/aspect_ratio, 0, 0, 0],
        [0, f, 0, 0], 
        [0, 0, (far+near)/(near-far), (2*far*near)/(near-far)],
        [0, 0, -1, 0]
    ], dtype=np.float32)
    
    return proj    

def create_mesh_buffers(render_mesh: shp.Mesh):
    """
    Create OpenGL buffers for a given mesh.
    
    Args:
        mesh (Mesh): The mesh object containing vertices and facets.
        
    Returns:
        tuple: A tuple containing the VAO, VBO, and EBO.
    """

    # VAO, VBO, end EBO Creation and Binding
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # Upload vertex data to GPU
    glBufferData(
        GL_ARRAY_BUFFER,
        render_mesh.vertices.astype(np.float32, copy= False).nbytes,
        render_mesh.vertices.astype(np.float32, copy= False),
        GL_STATIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 
                 render_mesh.facets.nbytes, 
                 render_mesh.facets, 
                 GL_STATIC_DRAW)

    # Link vertex data to shader layout (location = 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glEnableVertexAttribArray(0)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)
    if DEBUG:
        logger.info(f"Created VAO: {VAO}, VBO: {VBO}, EBO: {EBO}")
        logger.info(f"Vertex data type: {render_mesh.vertices.dtype}, facets data type: {render_mesh.facets.dtype}")
        logger.info(f"Vertex buffer size: {render_mesh.vertices.nbytes} bytes, facets buffer size: {render_mesh.facets.nbytes} bytes")
    # Return the created buffers
    return VAO, VBO, EBO

def get_all_uniform_locations(shader_program):
    """
    Get all uniform locations from the shader program.
    
    Args:
        shader_program (int): The OpenGL shader program ID.
        
    Returns:
        dict: A dictionary mapping uniform names to their locations.
    """
    num_uniforms = glGetProgramiv(shader_program, GL_ACTIVE_UNIFORMS)
    uniform_locations = {}
    
    for i in range(num_uniforms):
        name, size, type_ = glGetActiveUniform(shader_program, i)
        
        # Convert name to proper string if it's a numpy array
        if hasattr(name, 'tobytes'):
            name = name.tobytes().decode('utf-8').rstrip('\x00')
        elif isinstance(name, bytes):
            name = name.decode('utf-8').rstrip('\x00')
        elif hasattr(name, 'decode'):
            name = name.decode('utf-8').rstrip('\x00')
        # If it's already a string, use as-is
        
        location = glGetUniformLocation(shader_program, name)
        uniform_locations[name] = location
        
        if DEBUG:
            logger.info(f"Found uniform: '{name}' at location {location}")
        
    return uniform_locations # size, type_

def main():
    
    WIDTH, HEIGHT = 1260, 900
    ASPECT_RATIO = WIDTH / HEIGHT
    
    # Log system info
    if DEBUG:
        logger.info(" Logger Created, fetching system info...")
        sysinfo = get_system_info()
        for k, v in sysinfo.items():
            if k == "Resource_Limits":
                logger.info("=== Resource Limits ===")
                for limit_k, limit_v in v.items():
                    logger.info(f"  {limit_k}: {limit_v}")
            else:
                logger.info(f"{k}: {v}")

    # Initialize GLFW
    if not glfw.init():
        logger.error("Failed to initialize GLFW")
        return

    # Create window
    window = glfw.create_window(WIDTH, HEIGHT, "MeshViz - Hello OpenGL", None, None)
    if not window:
        logger.error("Failed to create GLFW window")
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glViewport(0, 0, WIDTH, HEIGHT)
    glfw.set_framebuffer_size_callback(window, framebuffer_size_callback)
    if DEBUG: logger.info("OpenGL context created")

    # Some Debugging Operations
    if DEBUG:
        version = glGetString(GL_VERSION).decode() # type: ignore
        renderer = glGetString(GL_RENDERER).decode() # type: ignore
        logger.info(f"OpenGL Version: {version}")
        logger.info(f"Renderer: {renderer}")

    shader_program = create_shader_program(BASIC_VERTEX_SHADER, BASIC_FRAGMENT_SHADER)
    
   # Create a tetrahedron and a cube mesh test objects
    size_test = 0.75
    Tet_mesh = shp.create_tetrahedron (size_test)
    Cube_mesh = shp.create_cube (size_test)

    RENDER_MESH = Cube_mesh  # Change this to Cube_mesh to render the cube instead

    VAO_1, VBO_1, EBO_1 = create_mesh_buffers(RENDER_MESH)
    
    UNIFORM_LOCS = get_all_uniform_locations(shader_program)
    if DEBUG:
        logger.info(f"Uniform locations: {UNIFORM_LOCS}")

    PROJ_LOC = UNIFORM_LOCS['projection']
    
    # Log vertex data upload
    if DEBUG:
        logger.info(f"Vertex data uploaded to GPU: {RENDER_MESH.vertices.shape[0]} vertices, {RENDER_MESH.facets.size} facets")
        logger.info(f"Vertex data type: {RENDER_MESH.vertices.dtype}, facets data type: {RENDER_MESH.facets.dtype}")
        logger.info(f"Vertex buffer size: {RENDER_MESH.vertices.nbytes} bytes, facets buffer size: {RENDER_MESH.facets.nbytes} bytes")

    # Link vertex data to shader layout (location = 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glEnableVertexAttribArray(0) 
    glBindVertexArray(0)

    # Main loop
    while not glfw.window_should_close(window):
        glClearColor(0.1, 0.1, 0.2, 1.0)  # Dark blue background
        glClear(GL_COLOR_BUFFER_BIT)
        
        glUseProgram(shader_program)
        glBindVertexArray(VAO_1)
        
        win_WIDTH , win_HEIGHT = glfw.get_window_size(window)
        ASPECT_RATIO = win_WIDTH / win_HEIGHT

        projection_matrix = create_orthographic_matrix(ASPECT_RATIO)
        glUniformMatrix4fv(PROJ_LOC, 
                           1, 
                           GL_FALSE, 
                           projection_matrix.astype(np.float32, copy=False))
        
        # Set mesh color based on the current mode
        mesh_type = 'cube' if RENDER_MESH == Cube_mesh else 'tetrahedron'
        set_mesh_color(UNIFORM_LOCS, RENDER_MESH, CURRENT_COLOR_MODE, mesh_type)
        
        glDrawElements(GL_TRIANGLES, # Type of geometry to draw based on the indices
                    RENDER_MESH.facets.size, # Number of indices to draw
                    GL_UNSIGNED_INT, None) # Type of indices and offset (None means start from the beginning)
        
        # glDrawArrays(GL_TRIANGLES, 0, Tet_mesh.vertices.shape[0]) # Alternative to draw without indices
        
        glBindVertexArray(0) # Unbinds the VAO for code safety

        glfw.swap_buffers(window)
        glfw.poll_events()

    # Cleanup GPU resources
    glDeleteVertexArrays(1, [VAO_1])
    glDeleteBuffers(1, [VBO_1])
    glDeleteBuffers(1, [EBO_1])
    glDeleteProgram(shader_program)

    # Terminate GLFW
    glfw.terminate()
    if DEBUG: logger.info("GLFW terminated")

if __name__ == "__main__":
    main()
    if DEBUG: logger.info("Correctly Exited Program")
