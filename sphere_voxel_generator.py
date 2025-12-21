import numpy as np
from stl import mesh

# 参数设置
diameter = 12
radius = diameter / 2
voxels = []

# 遍历寻找球体内的点
for x in range(diameter):
    for y in range(diameter):
        for z in range(diameter):
            # 计算体素中心与球心的距离
            center_dist = np.linalg.norm([x + 0.5 - radius, 
                                          y + 0.5 - radius, 
                                          z + 0.5 - radius])
            
            if center_dist <= radius:
                # 定义一个立方体的12个三角形面 (标准STL格式)
                # 这里简化处理，直接记录位置
                voxels.append([x, y, z])

# 定义立方体的8个顶点（相对于体素位置）
def get_cube_vertices(x, y, z):
    """返回立方体的8个顶点坐标"""
    return np.array([
        [x, y, z],           # 0: 左下前
        [x+1, y, z],         # 1: 右下前
        [x+1, y+1, z],       # 2: 右上前
        [x, y+1, z],         # 3: 左上前
        [x, y, z+1],         # 4: 左下后
        [x+1, y, z+1],       # 5: 右下后
        [x+1, y+1, z+1],     # 6: 右上后
        [x, y+1, z+1],       # 7: 左上后
    ])

# 将体素列表转换为集合，方便快速查找邻居
voxel_set = {tuple(v) for v in voxels}

# 定义每个面的三角形和对应的邻居检查方向
face_definitions = [
    # 前面 (z方向，检查 z-1)
    ([[0, 1, 2], [0, 2, 3]], (0, 0, -1)),
    # 后面 (z方向，检查 z+1)
    ([[4, 7, 6], [4, 6, 5]], (0, 0, 1)),
    # 左面 (x方向，检查 x-1)
    ([[0, 3, 7], [0, 7, 4]], (-1, 0, 0)),
    # 右面 (x方向，检查 x+1)
    ([[1, 5, 6], [1, 6, 2]], (1, 0, 0)),
    # 下面 (y方向，检查 y-1)
    ([[0, 4, 5], [0, 5, 1]], (0, -1, 0)),
    # 上面 (y方向，检查 y+1)
    ([[3, 2, 6], [3, 6, 7]], (0, 1, 0)),
]

# 收集所有外表面三角形
triangles = []

for voxel in voxels:
    x, y, z = voxel
    vertices = get_cube_vertices(x, y, z)
    
    # 检查每个面是否需要生成
    for face_triangles, neighbor_offset in face_definitions:
        # 检查这个方向是否有邻居
        neighbor_pos = (x + neighbor_offset[0], y + neighbor_offset[1], z + neighbor_offset[2])
        
        # 如果没有邻居，则生成这个面
        if neighbor_pos not in voxel_set:
            # 添加这个面的两个三角形
            for face_indices in face_triangles:
                v0 = vertices[face_indices[0]]
                v1 = vertices[face_indices[1]]
                v2 = vertices[face_indices[2]]
                triangles.append([v0, v1, v2])

# 创建STL网格数据
data = np.zeros(len(triangles), dtype=mesh.Mesh.dtype)
cube_mesh = mesh.Mesh(data)

# 填充三角形数据
for i, triangle in enumerate(triangles):
    cube_mesh.vectors[i] = np.array(triangle)

# 保存STL文件
output_file = 'sphere_voxel.stl'
cube_mesh.save(output_file)
print(f"成功生成STL文件: {output_file}")
print(f"体素数量: {len(voxels)}")
print(f"外表面三角形数量: {len(triangles)}")
