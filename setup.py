from setuptools import setup, find_packages

setup(
    name="electronic-analyzer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    author="HabitCanvas Team",
    author_email="info@habitcanvas.com",
    description="An AI-powered electronic component analysis system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/habitcanvas/electronic-analyzer",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "electronic-analyzer=run_electronic_analyzer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.html", "*.css", "*.js", "*.png", "*.svg"],
    },
)