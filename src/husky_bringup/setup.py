from glob import glob
from setuptools import setup

package_name = 'husky_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml') + glob('config/*.rviz')),
        ('share/' + package_name + '/scripts', glob('scripts/*.sh')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
        ('share/' + package_name + '/maps', glob('maps/*.png') + glob('maps/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='suzukh',
    maintainer_email='szkharuto0407@gmail.com',
    description='Launch files and configs for the Husky A200 (sim and real)',
    license='Apache-2.0',
)
