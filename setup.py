from setuptools import find_packages, setup
from glob import glob

package_name = 'dual_arm_ball_setter'

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
            'testrun_1 = dual_arm_ball_setter.testrun_1:main',
            'testrun_2 = dual_arm_ball_setter.testrun_2:main',
            'test_boundary = dual_arm_ball_setter.test_boundary:main',
            'testrun_tk = dual_arm_ball_setter.testrun_tk:main',
            'demo1 = dual_arm_ball_setter.demo1:main',
            'test_normal_task = dual_arm_ball_setter.test_normal_task:main',
            'final = dual_arm_ball_setter.final:main',
            'plotball = dual_arm_ball_setter.plotball:main',
            'plotcondition = dual_arm_ball_setter.plotcondition:main',
            'plotdata = dual_arm_ball_setter.plotdata:main',
        ],
    },
)
