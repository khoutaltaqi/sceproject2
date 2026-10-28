from setuptools import find_packages, setup

package_name = 'rm_flatness_nav'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        (
            'share/' + package_name,
            ['package.xml'],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='taqi',
    maintainer_email='taqi.khoutal@gmail.com',
    description='RoboMaster EP Core hardware checks',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'ep_check = rm_flatness_nav.ep_check:main',
            'floor_calibration = rm_flatness_nav.floor_calibration:main',
            'assignment_driver = rm_flatness_nav.assignment_driver:main',
            'flatness_planner = rm_flatness_nav.flatness_planner:main',
            'plot_results = rm_flatness_nav.plot_results:main',
        ],
    },
)
