from setuptools import setup

setup(
    name='iml',
    packages=['server'],
    include_package_data=True,
    install_requires=[
        'flask',
    ],
    entry_points={
        "console_scripts": [
            "iml-start = server.some_module:foo",
        ],
    },
)