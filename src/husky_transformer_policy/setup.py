from glob import glob
from setuptools import find_packages, setup
name = 'husky_transformer_policy'
setup(name=name, version='0.1.0', packages=find_packages(exclude=['test']),
      data_files=[('share/ament_index/resource_index/packages', ['resource/'+name]),
                  ('share/'+name, ['package.xml']), ],
      install_requires=['setuptools'], zip_safe=True,
      maintainer='alice', maintainer_email='xuyaoalice@gmail.com',
      description='Isolated Transformer policy interface for Husky',
      license='TODO: License declaration',
      entry_points={'console_scripts': ['policy_worker = husky_transformer_policy.policy_worker:main']})
