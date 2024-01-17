import numpy
import matplotlib.pyplot as plt
import trimesh


def construct_sphere_mesh(x=0, y=0, z=0) -> trimesh.Trimesh:
    """
    this function generate a mesh for a sphere, with the origin point at the center.
    """
    # Generate the sphere mesh
    radius = 0.2
    sections = 3
    mesh_sphere = trimesh.creation.icosphere(radius=radius, subdivisions=sections)
    # apply the translation
    mesh_sphere.apply_translation([x, y, z])
    
    return mesh_sphere



def construct_xyz_axis_mesh() -> trimesh.Trimesh:
    """
    this function generate a mesh for the xyz axis, with the origin point at the center.
    """
    # Generate the x axis mesh
    radius = 0.05
    height = 1
    sections = 20
    mesh_x_axis = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    mesh_x_axis.apply_transform(trimesh.transformations.rotation_matrix(numpy.pi/2, [0, 1, 0]))

    # Generate the y axis mesh
    radius = 0.05
    height = 1
    sections = 20
    mesh_y_axis = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    #mesh_y_axis.apply_translation([0, height/2, 0])
    mesh_y_axis.apply_transform(trimesh.transformations.rotation_matrix(numpy.pi/2, [0, 0, 1]))

    # Generate the z axis mesh
    radius = 0.05
    height = 1
    sections = 20
    mesh_z_axis = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    #mesh_z_axis.apply_translation([0, 0, height/2])
    mesh_z_axis.apply_transform(trimesh.transformations.rotation_matrix(numpy.pi/2, [1, 0, 0]))

    # combine the three mesh into an xyz axis
    mesh_xyz_axis = mesh_x_axis + mesh_y_axis + mesh_z_axis

    # export the xyz axis mesh as an obj file
    #mesh_xyz_axis.export("xyz_axis.obj")
    #print(mesh_xyz_axis.vertices[:, 2].min(), mesh_xyz_axis.vertices[:, 2].max())
    
    return mesh_xyz_axis


def construct_arrow_mesh(size: int) -> trimesh.Trimesh:
    """
    this function generate a mesh for an arrow, with the tip pointing towards the positive z direction to the origin point.
    size: the size of the arrow.
    """
    # Generate the cone mesh
    radius = 1.0 * size
    height = 1.5 * size
    sections = 20
    mesh_cone = trimesh.creation.cone(radius=radius, height=height, sections=sections)

    # Generate the cylinder mesh
    radius = 0.5 * size
    height = 5 * size
    sections = 20
    mesh_cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)

    # combine the two mesh into an arrow
    mesh_cone.apply_translation([0, 0, height/2])
    mesh_arrow = mesh_cylinder + mesh_cone
    mesh_arrow.apply_translation([0, 0, -height])

    # export the arrow mesh as an obj file
    #mesh_arrow.export("arrow.obj")
    #print(mesh_arrow.vertices[:, 2].min(), mesh_arrow.vertices[:, 2].max())
    
    return mesh_arrow
    
    
def move_the_arrow(mesh: trimesh.Trimesh, 
                   #origin: numpy.ndarray, 
                   target_position: numpy.ndarray,
                   target_direction: numpy.ndarray,
                   ) -> trimesh.Trimesh:
    """
    this function move the arrow mesh to the target point.
    """
    # compute the translation vector
    translation_vector = target_position #- origin
    # rotate the arrow to the target direction
    rotation_matrix = trimesh.transformations.rotation_matrix(
        trimesh.transformations.angle_between_vectors(numpy.array([0, 0, 1]), target_direction), 
        trimesh.transformations.vector_product(numpy.array([0, 0, 1]), target_direction)
        )
    
    # apply the rotation
    mesh.apply_transform(rotation_matrix)
    # apply the transformation
    mesh.apply_transform(trimesh.transformations.translation_matrix(translation_vector))

    return mesh


"""
使用方法：
1. 选择一个点的位置作为箭头的「注视点」，并且设定所需的箭头的方向：
    target_position = numpy.array([0, 0, 10])
    target_direction = numpy.array([-1, -1, -1])
2. 调用函数，生成一个指向该点的箭头：
    mesh = construct_arrow_mesh()
    mesh = move_the_arrow(mesh, target_position, target_direction)
3. 输出该箭头的 obj 文件：
    mesh.export("arrow.obj")
具体的例子参见下面的代码。
"""


if __name__ == "__main__":
    mesh = construct_arrow_mesh(size=0.5)
    target_position = numpy.array([0, 0, 10])
    target_direction = numpy.array([-1, -21, -1])
    mesh = move_the_arrow(mesh, target_position, target_direction)
    
    mesh = mesh + construct_xyz_axis_mesh() + construct_sphere_mesh(0, 0, 10)
    mesh.export("arrow.obj")
