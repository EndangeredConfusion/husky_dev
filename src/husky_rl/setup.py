from setuptools import find_packages, setup

package_name = 'husky_rl'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/models',
            ['models/dqn_eightsixmulti_lambda.zip']),
        ('share/' + package_name + '/launch',
            ['launch/rl.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alice',
    maintainer_email='xuyaoalice@gmail.com',
    description='RL policy and adaptive lambda nodes for Husky deployment',
    license='TODO: License declaration',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'rl_policy = husky_rl.rl_policy_node:main',
            'lambda_node = husky_rl.lambda_node:main',
        ],
    },
)
