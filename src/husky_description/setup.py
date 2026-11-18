from glob import glob
from setuptools import setup

package_name = 'husky_description'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf', glob('urdf/*.xacro')),
        ('share/' + package_name + '/meshes', glob('meshes/*.stl') + glob('meshes/*.dae')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='suzukh',
    maintainer_email='szkharuto0407@gmail.com',
    description='Extra sensors for the Husky A200 (URDF + Gazebo sensors)',
    license='Apache-2.0',
)
