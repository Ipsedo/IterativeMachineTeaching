from setuptools import setup

setup(
    name='iterative_machine_teaching',
    version='1.0.0',
    packages=[
        'iterative_machine_teaching'
    ],
    url='https://github.com/Ipsedo/IterativeMachineTeaching',
    license='',
    author='Ipsedo',
    author_email='',
    description='Iterative Machine Teaching implementation',
    entry_points={
        'console_scripts': [
            'iterative_machine_teaching = iterative_machine_teaching.__main__:main'
        ]
    },
    test_suite="tests"
)
