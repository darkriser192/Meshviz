import numpy as np
import sympy

# support_math functions will be integrated directly as needed placed in a separate file later in the development
# TODO Create functions to convert between affine, Quaternion, euler and vector-angle
# TODO Create a function that efficiently computes the transpose of a 4x4 affine transform
# TODO Create a function that efficiently computes the determinant of a 4x4 affine transform
# TODO optimize "import" statements across the codebase

#----- Global constants for program use -----#

# Engineering tolerance constants (all in mm²)
# Applied Nyquist (1/2) and FEA (1/3) safety factors to ensure adequate spatial sampling

# Base geometric scales
LASER_SPOT_DIAMETER = 0.05  # mm 
MELT_POOL_DIAMETER = 2 * LASER_SPOT_DIAMETER # mm
GEOMETRIC_TOLERANCE = 0.01  # mm (±0.01 mm mechanical tolerance)

# Area thresholds with safety factors
# Melt pool: equilateral triangle, diameter 0.1mm, with 1/3 safety factor
MELT_POOL_AREA_BASE = (np.sqrt(3)/4) * (MELT_POOL_DIAMETER * np.sqrt(3))**2  # ≈ 1.3e-2 mm²
MELT_POOL_AREA_THRESHOLD = MELT_POOL_AREA_BASE / 3  # ≈ 4.3e-3 mm²

# Geometric tolerance: equilateral triangle, side 0.01mm, with 1/3 safety factor  
GEOMETRIC_TOLERANCE_AREA_BASE = (np.sqrt(3)/4) * GEOMETRIC_TOLERANCE**2  # ≈ 4.33e-5 mm²
GEOMETRIC_TOLERANCE_AREA = GEOMETRIC_TOLERANCE_AREA_BASE / 2  # ≈ 2.17e-5 mm²

# Numerical precision: well above float64 machine epsilon but practically zero
NUMERICAL_PRECISION_AREA = 1e-12  # mm²

# Normal Degeneracy
# TODO consutrct some degeneracy for the Normal or angular tollerance

#----- Support functions for geometric calculations -----#
# Degeneracy classification thresholds
def classify_triangle_degeneracy(area):
    """
    Classify triangle degeneracy levels for different use cases.
    
    Args:
        area: Triangle area in mm²
        
    Returns:
        dict: Boolean flags for different degeneracy levels
    """
    return {
        'critically_degenerate': area < NUMERICAL_PRECISION_AREA,
        'thermally_insignificant': area < MELT_POOL_AREA_THRESHOLD, 
        'visually_insignificant': area < GEOMETRIC_TOLERANCE_AREA
        # TODO 'normal_degeneracy' - wondering if this is meaningfull

    }

def compute_facet_area(vector1: np.ndarray, # A structure of vectors representing the first vector of the triangle. Shape (N,3)
                       vector2: np.ndarray # A structure of vectors representing the second vector of the triangle Shape (N,3)
                       ) -> np.ndarray:
    """
    Assumes inputs are already float64
    Takes the XYZ coordinates of 3 points and computes the area of the triangle shape
    We have abandoned the idea of computing the area of a quadrilateral, as that can be 
    simplified as 2 triangles and most of our work is expected to be triangular meshes
    """
    return (0.5 * np.linalg.norm(np.cross(vector1,vector2), axis = 1))
    
def compute_facet_normal(vector1: np.ndarray, # A structure of vectors representing the first vector of the triangle
                         vector2: np.ndarray # A structure of vectors representing the second vector of the triangle
                         ) -> np.ndarray:
    """
    Assumes inputs are already float64
    Given 2 vectors normal of those faces. returns the normalized vector
    """

    # TODO: Implement normal magnitude-based degeneracy detection
    # Concept: When ||cross_product|| approaches zero, the triangle becomes degenerate
    # even if area calculation might still be numerically stable.
    # 
    # Implementation ideas:
    # 1. Define normal magnitude threshold (e.g., NORMAL_MAGNITUDE_THRESHOLD = 1e-10)
    # 2. Flag triangles where norm < threshold as "normal_degenerate"
    # 3. This could catch cases where area calculation succeeds but normal is unreliable
    # 4. Potential optimization: Skip expensive normalize operations for degenerate normals
    # 5. Consider relationship between area threshold and normal magnitude threshold
    #    (they should be mathematically consistent: norm ≈ 2*area for triangle)
    #
    # Benefits: 
    # - Early detection of problematic triangles
    # - Skip unnecessary normalization computations
    # - More robust geometric algorithms downstream

    # Compute the values used for the normals
    cross = np.cross(vector1,vector2)
    norm = np.linalg.norm(cross, axis = 1)

    # Protect against zero-length cross products
    zero_mask = norm < NUMERICAL_PRECISION_AREA**0.5  # sqrt of area threshold
    norm[zero_mask] = 1.0  # Avoid division by zero

    normal = cross/norm[:, np.newaxis] 
    normal[zero_mask] = [0.0, 0.0, 0.0]  # Set degenerate normals to zero

    return normal

def compute_vertex_normal(facet_normals: np.ndarray, # A structure of vectors representing the theretical normal at the vertex location
                          facets: np.ndarray,
                          areas: np.ndarray,
                          vertex_normals: np.ndarray): #-> np.ndarray:
    """
    Computes vertex normals by averaging facet normals weighted by facet area.
    This function assumes facet_normals and facets are already computed.

    Args:

        facet_normals: np.ndarray

        
    Returns:
        In place update vertex_normals: np.ndarray
    """
    # TODO needs to be implemented weigthed average of the normals of each faces to comput the normal vectors for a vertex
    pass

def compute_center_points(points):
    """
    Computes centroid of points with flexible input handling.
    
    Args:
        points: np.ndarray 
            - Shape (N, 3): N points → returns (3,) centroid
            - Shape (M, N, 3): M groups of N points → returns (M, 3) centroids
    
    Returns:
        np.ndarray: Centroid coordinates
    """
    if points.ndim == 2:
        return np.mean(points, axis=0)  # Single group: average across points
    elif points.ndim == 3:
        return np.mean(points, axis=1)  # Multiple groups: average across points in each group
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {points.shape}")

def _generate_tetrahedron_geometry(size = 1.0):
    """
    Takes a dimension and creates a tetrahedron that can be inscribed inside a sphere of radious 
    equal to the size
    """
    a = size * np.sqrt(8/9)  # Distance from center to face centers
    b = size * np.sqrt(2/9)  # Distance in xy-plane
    c = size / 3             # Height offset
    
    v0 = np.array([0.0, 0.0, size])                    # Top vertex
    v1 = np.array([a, 0.0, -c])                        # Bottom vertex 1
    v2 = np.array([-b,  np.sqrt(2/3)*size, -c])         # Bottom vertex 2  
    v3 = np.array([-b, -np.sqrt(2/3)*size, -c])        # Bottom vertex 3
    vertices = np.array([v0,v1,v2,v3])
    facets = np.array([
        [0,1,2],
        [0,2,3],
        [0,3,1],
        [1,3,2]])
    return vertices, facets

def _generate_cube_geometry(size = 1.0):
    # Decided to leave winding as it is to debug the normals when rendering is implemented
    """
    Generates a cube with vertices scaled by the given size.
    Vertices are centered at the origin, with size defining the half-length of each edge.
    
    """
    # Vertices of a cube centered at the origin
    v0 = np.array([ 0.5,  0.5,  0.5]) # top 1 vertex num 1
    v1 = np.array([-0.5,  0.5,  0.5]) # top 2 vertex num 2
    v2 = np.array([ 0.5, -0.5,  0.5]) # top 3 vertex num 3
    v3 = np.array([-0.5, -0.5,  0.5]) # top 4 vertex num 4
    v4 = np.array([ 0.5,  0.5, -0.5]) # bottom 1 vertex num 5
    v5 = np.array([-0.5,  0.5, -0.5]) # bottom 2 vertex num 6
    v6 = np.array([ 0.5, -0.5, -0.5]) # bottom 3 vertex num 7
    v7 = np.array([-0.5, -0.5, -0.5]) # bottom 4 vertex num 8

    vertices = np.array([v0, v1, v2, v3, v4, v5, v6, v7]) * size
    facets = np.array([
        [0,1,3],
        [1,3,2],
        [0,4,5],
        [5,1,0],
        [1,5,6],
        [6,3,1],
        [2,7,6],
        [7,2,3],
        [4,7,5],
        [5,0,4],
        [4,6,5],
        [5,6,7]
    ])

    return vertices, facets

def _generate_sphere_geomery(redius = 1.0, resolution = 1):
    # TODO implement the sphere generation. Resolution defines the number of segments in the sphere. 1 generates an octahedron, 2 generates a dodecahedron, etc.
    
    pass

def _generate_triangle_geometry(size = 1.0):
    """
    Generates a triangle with vertices scaled by the given size.
    Vertices are centered at the origin
    """
    # Example Triangle
    vertices = np.array([
        [ 0.0,  0.5, 0.0],  # Top
        [-0.5, -0.5, 0.0],  # Bottom Left
        [ 0.5, -0.5, 0.0]   # Bottom Right
    ], dtype = np.float64) * size  # Scale by size

    facets = None  # No facets needed for a single triangle, but can be added later if needed

    return vertices, facets

def _gererate_circle_geometry(radius = 1.0, resolution = 4):
    """
    Generates a circle with vertices scaled by the given radius.
    Vertices are centered at the origin.
    resolution defined the number of segments in the circle

    Args:
        radius: Radius of the circle
        resolution: Number of segments in the circle
    Returns:
        np.ndarray: Vertices of the circle
    """
    assert resolution >= 4, "Resolution must be at least 4 to form a circle"

    angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = np.zeros_like(x)  # All points lie in the XY plane
    
    # Stack coordinates into a single array
    vertices = np.column_stack((x, y, z)) # Shape (resolution, 3)

    # Form triangles by connecting the center to each segment
    center = np.array([[0.0, 0.0, 0.0]])  # Center point at the origin
    indices = np.arange(resolution)
    # TODO implement facet generation
    # Method: Create N triangles from center to circumference
    # Pattern: for each segment i, create triangle [center_index, i, (i+1) % resolution]
    # center_index = resolution (last vertex), circumference vertices = 0 to resolution-1
    # Example for resolution=6: triangles = [[6,0,1], [6,1,2], [6,2,3], [6,3,4], [6,4,5], [6,5,0]]
    facets = None
    
    return vertices, facets

def _generate_cylinder_geometry(radius = 1.0, length = 1.0, resolution = 1):
    # TODO implement the cylinder generation
    """
    Generates a cylinder with vertices scaled by the given radius and length.
    Vertices are centered at the origin.
    resolution defines the number of segments in the cylinder
    Args:
        radius: Radius of the cylinder
        length: Length of the cylinder
        resolution: Number of segments in the cylinder
    
    Returns:
        np.ndarray: Vertices of the cylinder
        np.ndarray: Facets of the cylinder
    """
    vertices, lid_facets = _gererate_circle_geometry(radius, resolution)  # Get circle vertices adn facets
    top_vertices = vertices.copy() + length  # Copy circle vertices for top lid
    bottom_vertices = vertices.copy() - length  # Copy circle vertices for bottom lid
    
    # TODO implement facets generation for both lids and the cylinder sides
    facets = None  # Placeholder for facets, needs implementation


    return vertices, facets  # Return vertices and facets of the cylinder

def transform_vector(trasnform: np.ndarray, vector: np.ndarray) -> np.ndarray:
    # TODO Create Transform_vector support function. Efficient multiplication of a 4x4 afine transformation and a 3,1 vector
    """
    Applies a 4x4 affine transformation to a 3D vector.
    Args:
        transform: 4x4 affine transformation matrix
        vector: 3D vector to transform (shape (3,1) or (N, 3))
    Returns:
        np.ndarray: Transformed vector(s) in homogeneous coordinates (shape (3,1) or (N, 3))
    """
    result_vector = np.zeros((4,1))  # Placeholder for the result

    return result_vector

def transform_transform(transform_mod: np.ndarray, transform_target: np.ndarray) -> np.ndarray:
    # TODO Create Transform_Transform support function. Efficient multiplication of a 4x4 afine transformation and a 4,4 affine transform
    """
    Combines two affine transformations into a single transformation.
    """

    result_transform = np.eye(4)  # Placeholder for the result
    return result_transform

def inverse_Homogeneous_transform(transform): #Must be a Trasnform class object
    # TODO Create a function that efficiently computes the inverse of a 4x4 affine transform
    """
    Computes the inverse of a 4x4 affine transformation matrix.
    """
    inverse_rotation_matrix = np.eye(3)  # Placeholder for the rotation component
    inverse_translation_vector = np.zeros((3,1))  # Placeholder for the translation component
    
    return inverse_rotation_matrix, inverse_translation_vector

class Transform():
    """
    Class to represent a 4x4 affine transformation matrix.
    This class provides methods for efficient multiplication with vectors and other transformations.
    """
    def __init__(self):
        self.rotation = np.eye(3, dtype=np.float64)
        self.translation = np.zeros((3,1), dtype=np.float64)
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # Identity quaternion
        self.euler_angles = np.zeros((3,), dtype=np.float64)  # Euler angles in radians
        self.inverse_rotation = np.eye(3, dtype=np.float64)
        self.inverse_translation = np.zeros((3,1), dtype=np.float64)
        self.parent = 0  # Index of the parent transformation, if 0 then it is the world transformation

    def as_homogeneous_matrix(self):
        """
        Converts the transformation to a 4x4 affine matrix incase it is needed
        """
        pass
    
    def as_euler_vecor(self):
        """
        Converts the transformation to a 3D vector representing the euler angles plus trasnlation
        """
        
        pass

    def as_quaternion(self):
        """
        Converts the transformation to a quaternion representation plus trasnlation
        """

        pass

    def get_rotation_matrix(self):
        return self.rotation
    
    def get_translation_vector(self):
        return self.translation
    
    def get_quaternion(self):
        return self.quaternion
    
    def get_euler_angles(self):
        return self.euler_angles
    
    def get_inverse_rotation_matrix(self):
        if self.inverse_rotation is None or np.any(self.inverse_rotation == 0):
            self.inverse_rotation, self.inverse_translation = inverse_Homogeneous_transform(self)


#----- Objects and classes for the Meshviz library -----#
## Mesh object that contains optimized triangular infomration and computes any other meaningfull data
class Mesh(): 
    """
    Fundamental object storing vertices, facets, and related data.
    """
    # TODO implement a default color if no color is provided
    # TODO implement a way to read color for everything the same if only one color is provided
    # TODO implement a sparse structure that applies a none default color to every vertex that has not been specified
    
    # TODO: Implement sparse degeneracy storage to reduce memory overhead
    # Current issue: Storing degeneracy flags for every triangle wastes memory
    # 
    # Approach 1: Sparse dictionary/set approach
    # degeneracy_map = {
    #     facet_index: {'critically_degenerate': True, 'thermally_insignificant': False, ...},
    #     ...
    # }
    # 
    # Approach 2: Separate arrays per degeneracy type
    # critically_degenerate_indices = np.array([...])  # Only store indices of degenerate triangles
    # thermally_insignificant_indices = np.array([...])
    # 
    # Approach 3: Bitfield approach (Python enum + integer storage)
    # from enum import IntFlag
    # class DegeneracyType(IntFlag):
    #     NONE = 0
    #     CRITICALLY_DEGENERATE = 1
    #     THERMALLY_INSIGNIFICANT = 2  
    #     VISUALLY_INSIGNIFICANT = 4
    #     NORMAL_DEGENERATE = 8
    #
    # IntFlag approach (values can be combined with bitwise operations)
    # class DegeneracyFlags(IntFlag):
    #     NONE = 0
    #     AREA_DEGENERATE = 1
    #     NORMAL_DEGENERATE = 2
    #     THERMALLY_INSIGNIFICANT = 4
    #     VISUALLY_INSIGNIFICANT = 8
    #
    # degeneracy_flags = np.array([DegeneracyType.NONE, DegeneracyType.CRITICALLY_DEGENERATE | DegeneracyType.NORMAL_DEGENERATE, ...])
    #
    # Memory comparison:
    # - Current: 4 bools × 4 bytes × N triangles = 16N bytes
    # - Sparse: ~8 bytes × (degenerate triangles only) = 8D bytes where D << N
    # - Bitfield: 1 int × 4 bytes × N triangles = 4N bytes (75% reduction)

    def __init__(
        self, 
        vertices = None,  # [x, y, z] vertex position we have moved the color into its own location
        parent = 0, # Index of the parent trasnfomration. If 0 then it is the root transformation
        transform = None, # Transformation between the parent trasnformation and all the coordinates of the vertices
        facets = None, # Defines the facets of the mesh as triplets of the vertex indexes
        color = None, # RGBA values of the color of the material, if non asigned, single color for every vertex
        normals = None # Facet normals in order of how they appear
    ):
        self.parent = int(parent)
        self.transform = (np.eye(4, dtype=np.float64) if transform is None else np.asarray(transform, dtype=np.float64))
        self.vertices = (np.empty((0,3), dtype=np.float64) if vertices is None else np.asarray(vertices, dtype=np.float64))
        self.facets = (np.empty((0,3), dtype=np.uint32) if facets is None else np.array(facets, dtype=np.uint32)) # there can be no "negative" indexing of faces so uint should be more memory efficient
        # TODO implement a material definition
        self.solid_color = np.array([0.5, 0.5, 0.5, 1.0], dtype=np.float32) # RGBA Default color, if no color is provided
        
        # Initialize derived properties
        self.num_facets = len(self.facets)
        self.num__vertex = len(self.vertices)

        # Preallocate normals with zeros
        self.facet_normals = np.zeros((self.num_facets, 3), dtype=np.float64) if normals is None else np.asarray(normals, dtype=np.float64)
        self.vertex_normals = np.zeros((self.num__vertex, 3), dtype=np.float64)
        self.facets_area = np.zeros((self.num_facets, 1), dtype=np.float64)
        self.facet_centers = np.zeros((self.num_facets, 3), dtype=np.float64)
        self.vertex_color = np.zeros((self.num__vertex, 4), dtype=np.float32) if color is None else np.asarray(color, dtype=np.float32) # RGBA color for each vertex, if not provided then it is a solid color

        # Validate shapes
        if self.vertices.ndim != 2 or self.vertices.shape[1] < 3:
            raise ValueError(f"Vertices array must have shape (N, 3), got {self.vertices.shape}")
        if self.facets.ndim != 2 or self.facets.shape[1] != 3:
            raise ValueError(f"Facets array must have shape (M, 3), got {self.facets.shape}")
        if self.facets.size and (self.facets.min() < 0 or self.facets.max() >= self.num__vertex):
            raise IndexError("Facet indices out of range")

        # Validate index ranges
        if np.any(self.facets < 0) or np.any(self.facets >= len(self.vertices)):
            raise IndexError("Facet indices are out of range for given vertices")

    def update_facet_area(self, update_degeneracy: bool = False):
        v0 = self.vertices[self.facets[:,0]]
        v1 = self.vertices[self.facets[:,1]]
        v2 = self.vertices[self.facets[:,2]]
        
        self.facets_area = compute_facet_area(v1-v0, v2-v0).reshape(self.num_facets, 1)

        critically_degenerate_mask = self.facets_area < NUMERICAL_PRECISION_AREA
        self.facets_area[critically_degenerate_mask] = 0.0

        # TODO Create a update degeneracy flag for this function
        
    def update_facet_normal(self, update_degeneracy: bool = False): # changed calc with update to diferentiate from outsider helpers
        """
        Computes facet normals using the global compute_face_normal() function.
        Stores result in self.face_normals.
        """
        v0 = self.vertices[self.facets[:,0]]
        v1 = self.vertices[self.facets[:,1]]
        v2 = self.vertices[self.facets[:,2]]
        
        self.facet_normals = compute_facet_normal(v1-v0, v2-v0)
    
    def update_facet_center(self):
        """
        Vectorized computation of facet centers using advanced indexing.
        Computes centers for all facets simultaneously.
        """
        self.facet_centers = compute_center_points(self.vertices[self.facets])
        pass

    def update_vertex_normal(self): # changed calc with update to diferentiate from outsider helpers
        """
        Computes vertex normals using the global compute_vertex_normal() function.
        Requires self.facet_normals (will auto-generate if missing).
        """
        if not np.any(self.facet_normals):  # If facet_normals all zero
            self.update_facet_normal()
        
        compute_vertex_normal(self.facet_normals, self.facets, self.vertex_normals) # type: ignore

    def update_structures(self, update_degeneracy: bool = False):
       # TODO Update -> given that the inputs to compute_facer_normal and compute_facet_area are indentical, we can sdave computation time by computing the vectors once and feeding them to the functions
        v0 = self.vertices[self.facets[:,0]]
        v1 = self.vertices[self.facets[:,1]]
        v2 = self.vertices[self.facets[:,2]]

        self.facets_area = compute_facet_area(v1-v0, v2-v0).reshape(self.num_facets, 1)
        self.facet_normals = compute_facet_normal(v1-v0, v2-v0)

        # TODO Create a update degeneracy flag for this function

    @classmethod
    def from_tetrahedron(cls, size=1.0):
        return create_tetrahedron(size)  # Delegates to factory function
    
    @classmethod
    def from_cube(cls, size = 1.0):
        return create_cube(size) # Delegates to factory function

    @classmethod
    def from_sphere(cls, radius=1.0, resolution=8):
        return create_sphere(radius, resolution) # Delegates to factory function

#-----------------------------------------------------------------------------#
# Standalone function that returns a teteahedron mesh 
def create_tetrahedron(size=1.0) -> Mesh:
    vertices, facets = _generate_tetrahedron_geometry(size)
    return Mesh(vertices=vertices, facets=facets)

def create_cube(size = 1.0) -> Mesh:
    vertices, facets = _generate_cube_geometry(size)
    return Mesh(vertices=vertices, facets=facets)

def create_sphere(size = 1.0, resolution = 1.0) -> Mesh:
    # TODO Implement
    return Mesh()

# For contour generation
# TODO This fucntion requires more context to be implemented
# It should return the triangles that are above the geometric tolerance for contour splines
# This will be used to generate contour splines for the mesh
def get_contour_triangles(mesh: Mesh):
    """Only use triangles above geometric tolerance for contour splines."""
    valid_mask = mesh.facets_area[:, 0] >= GEOMETRIC_TOLERANCE_AREA
    return mesh.facets[valid_mask]

"""
# For thermal simulation  
def get_thermally_significant_triangles(mesh):
    '''Only use triangles above melt pool threshold for thermal analysis.'''
    valid_mask = mesh.facets_area[:, 0] >= MELT_POOL_AREA_THRESHOLD
    return mesh.facets[valid_mask]
"""

# For rendering (use everything that's not critically degenerate)
def get_renderable_triangles(mesh):
    """Use all triangles except critically degenerate ones."""
    valid_mask = mesh.facets_area[:, 0] >= NUMERICAL_PRECISION_AREA
    return mesh.facets[valid_mask]

## SDF object that contains optimized metric information for signed distance functions
# TODO Implement SDF object that contains optimized metric information for signed distance functions
# This will be used to compute the distance from a point to the surface of the mesh aswell for 3D volume rendering
class SDF():
    # TODO Implement
    pass



# Contour class that represents a contour spline
# This will be used to represent the contour of the mesh in 2D space. Either as a 2D projection or as a 3D contour.
class Contour():
    """
    Class to represent a contour spline.
    This will be used to represent the contour of the mesh in 2D space. Either as a 2D projection or as a 3D contour.
    """
    # TODO Implement
    pass

# Loads a mesh file unto a mesh object
def file2mesh(file_path: str) -> Mesh:
    """
    Reads a type of mesh file and load the data into a  unified Mesh class.
    This mesh data class contains a unified set of data that either auto computes upon 
    load or from the source file. 
     
    Supproted mesh files:
        STL, ACII or Binary -> 1st implementation
        3MF -> 2nd implementation
        OBJ -> 3rd implementation
    
    Args:
        file_path: location of the mesh file, if empty creates a unit circle with
        16 triangles
    
    Returns:
        Mesh: Returns a Mesh class object with the minimal amount amount of infomration
        provided by the sorce files (if an STL does not have vertex normals then it keeps
        that field empty, or if id does not provide color/ material it returns a default
        value)
    """
    # TODO: implement STL reader, needs a way to diferentiate between binary and ascii STL files
    # TODO: implement 3MF reader
    # TODO: implement OBJ reader
    
    return Mesh()

def ContourFromMesh(intersection_plane: None, # TODO implement intersection plane
                    mesh: Mesh,
                    resolution =  0.1) -> Contour:
    """
    Generates a contour spline from a mesh object.
    Args:
        mesh: Mesh object to generate contour from
        resolution: Resolution of the contour spline
        intersection_plane: Plane to intersect with the mesh
        
    Returns:
        Contour: Contour spline object
    """
    # TODO Implement contour generation from mesh
    return Contour()

def ContourFromSDF(intersection_plane: None, # TODO implement intersection plane
                   sdf: SDF,
                   resolution: float = 0.1,
                   threshold: float = GEOMETRIC_TOLERANCE_AREA) -> Contour:
    """
    Generates a contour spline from a signed distance function (SDF).
    
    Args:
        sdf: SDF object to generate contour from
        resolution: Resolution of the contour spline
        intersection_plane: Plane to intersect with the SDF
        
    Returns:
        Contour: Contour spline object
    """
    # TODO Implement contour generation from SDF
    return Contour()

def test_convex_shape_winding(mesh: Mesh):
    """
    Test if normals point outward for any convex shape centered at origin.
    
    Principle: For convex shapes, normal should point in same direction as 
    vector from shape center to face center.
    """
    # Ensure normals are computed
    if not np.any(mesh.facet_normals):
        mesh.update_facet_normal()

    # Ensure centers are computed
    if not np.any(mesh.facet_centers):
        mesh.update_facet_center()
    
    # Shape center (centroid of all vertices)
    shape_center = compute_center_points(mesh.vertices)

    outward_directions = mesh.facet_centers - shape_center

    dot_products =np.sum(mesh.facet_normals * outward_directions, axis=1)
    
    return dot_products

#----- Main function to test the Meshviz - Shapes library -----#
def main():
    v1_single = np.array([[1, 0, 0]])  # Shape: (1, 3)
    v2_single = np.array([[0, 1, 0]])  # Shape: (1, 3)
    v1_multi = np.array([[1, 0, 0],
                         [2, 0, 0]])  # Shape: (2, 3)
    v2_multi = np.array([[0, 1, 0], 
                         [0, 2, 0]])  # Shape: (2, 3)
    # Single triangle area test
    area = compute_facet_area(v1_single, v2_single)
    print("Single triangle area:", area == 0.5)  # Should be [0.5]

    # Multiple triangles area test  
    areas = compute_facet_area(v1_multi, v2_multi)
    print("Multiple triangle areas:", areas == [0.5,2.0])  # Should be [0.5, 2.0]

    # Single triangle normals
    normals = compute_facet_normal(v1_single, v2_single)
    print("Single triangle normal:", normals)
    # Multiple triangle normals
    normals = compute_facet_normal(v1_multi, v2_multi)
    print("Multiple triangle normal:", normals)

    size = 3
    tet_1 = create_tetrahedron(size)
    tet_1.update_facet_area()
    tet_1.update_facet_normal()
    tet_1.update_facet_center()

    print(f"Tetrahedron vertices shape: {tet_1.vertices.shape}")  # Should be (4, 3)
    print(f"Tetrahedron facets shape: {tet_1.facets.shape}")     # Should be (4, 3)
    print(f"Tetrahedron vertices:")
    print(tet_1.vertices)

    # Verify all vertices are on sphere of radius 2.0
    distances = np.linalg.norm(tet_1.vertices, axis=1)
    print(f"Distances from origin: {distances}")  # Should all be ~2.0
    print(f"All on sphere? {np.allclose(distances, size)}")

    # Unit test for simple convex shapes centered at the origin
    # This tests if the normals on a mesh point in the outward direction
    print("Test Normals")
    dot_products = test_convex_shape_winding(tet_1)
    print(dot_products)

    cube_1 = create_cube(size)
    print(f"Cube vertices shape: {cube_1.vertices.shape}")
    print(f"Cube facets shape: {cube_1.facets.shape}")
    print(f"Cube vertices:")
    print(cube_1.vertices)

    # Test for simple convex shapes centered at the origin
    # This tests if the normals on a mesh point in the outward direction
    print("Cube Normals")
    dot_products = test_convex_shape_winding(cube_1)
    print(dot_products)

if __name__ == "__main__":
    main()
#----- End of Meshviz - Shapes library -----#