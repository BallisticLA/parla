import setuptools

setuptools.setup(
    name='rlapy',
    version='0.1.1',
    author='',
    url='https://github.com/rileyjmurray/rlapy',
    author_email='',
    description='Randomized linear algebra in python',
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
