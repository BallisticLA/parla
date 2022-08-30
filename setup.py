import setuptools

setuptools.setup(
    name='parla',
    version='0.1.5',
    author='',
    url='https://github.com/BallisticLA/parla',
    author_email='rjmurray@berkeley.edu',
    description='Python Algorithms for Randomized Linear Algebra',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        'Development Status :: 2 - Pre Alpha',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    python_requires='>=3.7',
    install_requires=["numpy >= 1.17",
                      "scipy >= 1.1",
                      "matplotlib",
                      "pytest",
                      "pandas"]
)
