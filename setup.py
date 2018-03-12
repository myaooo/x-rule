from setuptools import setup

setup(
    name='iml',
    packages=['server'],
    include_package_data=True,
    install_requires=[
        'flask==0.12',
        'scikit-learn>=0.19',
        'numpy==1.14',
        'gunicorn==19.7.1',
        'dill>=0.2.7',
        'flask-cors>=3.0.3',
        'scipy>=1.0.0',
        'xlrd>=0.9.0',
        'pandas>=0.22.0',
    ],
    entry_points={
        "console_scripts": [
            "iml-start = server.some_module:foo",
        ],
    },
)