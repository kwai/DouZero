import setuptools

VERSION = '1.0.0'

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="douzero",
    version=VERSION,
    author="Kuai",
    author_email="daochen.zha@tamu.edu",
    description="DouZero DouDizhu AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    license='Apache License 2.0',
    keywords=["DouDizhu", "RL"],
    packages=setuptools.find_packages(),
    install_requires=[
        'torch',
    ],
    requires_python='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: Apache 2.0 License",
        "Operating System :: OS Independent",
    ],
)
