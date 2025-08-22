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

    # VAO and VBO Creation and Binding
    VAO_1 = glGenVertexArrays(1) # 1 creates 1 buffer, 2 returns an array of 2 buffers
    glBindVertexArray(VAO_1)
    proj_loc = glGetUniformLocation(shader_program, "projection")

    VBO_1 = glGenBuffers(1) # 1 creates 1 buffer, 2 returns an array of 2 buffers
    glBindBuffer(GL_ARRAY_BUFFER, VBO_1)

    EBO_1 = glGenBuffers(1) # 1 creates 1 buffer, 2 returns an array of 2 buffers
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_1)

    # Create a tetrahedron and a cube mesh test objects
    size_test = 0.75
    Tet_mesh = shp.create_tetrahedron (size_test)
    Cube_mesh = shp.create_cube (size_test)

    RENDER_MESH = Cube_mesh  # Change this to Cube_mesh to render the cube instead
    
    # Upload vertex data to GPU
    glBufferData(
        GL_ARRAY_BUFFER,
        RENDER_MESH.vertices.astype(np.float32, copy= False).nbytes,
        RENDER_MESH.vertices.astype(np.float32, copy= False),
        GL_STATIC_DRAW)
    
    # Upload facet data to GPU
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        RENDER_MESH.facets.nbytes,
        RENDER_MESH.facets,
        GL_STATIC_DRAW)
    
    color_loc = glGetUniformLocation(shader_program, "meshColor")
    
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
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_matrix.astype(np.float32, copy=False))
        
        glUniform4fv(color_loc, 1, RENDER_MESH.solid_color)  # Set the mesh color
        
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
