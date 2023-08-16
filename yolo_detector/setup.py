from setuptools import setup

package_name = 'yolo_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dknt',
    maintainer_email='963725175@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'service = yolo_detector.service_test:main',
            'client = yolo_detector.test_client:main',
            'detect_server = yolo_detector.detect_server:main',
            'detect_client = yolo_detector.detect_client:main',
        ],
    },
)
