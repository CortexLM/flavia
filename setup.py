from setuptools import setup, find_packages

# Read the contents of your requirements.txt file
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='bittensor_subnet',  # Replace with your project's name
    version='0.0.0',  # Replace with your project's version
    author='Your Name',  # Replace with your name
    author_email='your.email@example.com',  # Replace with your email
    description='A short description of your Bittensor subnet project',  # Provide a short description
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourgithubprofile/your_project',  # Replace with the URL to your project
    packages=find_packages(),
    install_requires=required,  # List of requirements read from requirements.txt
    classifiers=[
        # Choose classifiers from https://pypi.org/classifiers/ to describe your project
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',  # Specify the minimum version of Python required
)