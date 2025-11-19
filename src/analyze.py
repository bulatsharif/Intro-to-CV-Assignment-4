from plyfile import PlyData
import numpy as np

plydata = PlyData.read('input-data/Theater_uncompressed.ply')

print(f"Elements: {[e.name for e in plydata.elements]}")

vertex_element = plydata['vertex']
print(f"\nProperties available")
print(vertex_element.properties)

x = vertex_element['x']
y = vertex_element['y']
z = vertex_element['z']

print(f"\nScene Bounds:")
print(f"X: {np.min(x)} to {np.max(x)}")
print(f"Y: {np.min(y)} to {np.max(y)}")
print(f"Z: {np.min(z)} to {np.max(z)}")
print(f"Total Gaussians: {len(x)}")

