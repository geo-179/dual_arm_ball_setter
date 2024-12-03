from setuptools import find_packages, setup
from glob import glob

package_name = 'dual-arm-ball-setter'

other_files = [
    ('share/' + package_name + '/launch', glob('launch/*')),
    ('share/' + package_name + '/rviz',   glob('rviz/*')),
    ('share/' + package_name + '/urdf',   glob('urdf/*')),
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ]+other_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'testrun = dual-arm-ball-setter.testrun:main',
        ],
    },
)
