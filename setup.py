from setuptools import setup, find_packages

def requirements():
    with open(r'requirements.txt') as f:
        required = f.read().splitlines()
    return required

def readme():
    with open('README.md') as f:
        README = f.read()
        return README

setup(
        name = 'ResumeParser',
        version = '1.0.0',
        description = 'It a simple package for training and classification of resumes.',
        long_description = readme(),
        long_description_content_type = 'text/markdown',
        url = "https://github.com/shreyas2306/ResumeParser.git",
        author = "Shreyas Nanaware/Krushna Kumar Nalange",
        license = "GNU General Public License v3.0",
        classifiers = [
            "License :: OSI Approved :: GNU General Public License v3.0",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",

        ],
        packages = find_packages(),
        include_package_data =  True,
        package_data = {'ResumeParser': ['model/*']},
        install_requires = requirements()

    )