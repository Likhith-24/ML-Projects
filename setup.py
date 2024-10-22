from setuptools import setup, find_packages
from typing import List

hyphen_e_dot = '-e .' # Define a string with hyphen, e and dot
def get_requirements(file_path:str) -> List[str]: # Define a function to read the requirements from the file
    '''This Function reads the requirements from the file and returns a list of requirements'''

    requirements=[] # Initialize an empty list to store the requirements
        
    with open(file_path) as file_obj: # Open the file in read mode
        requirements=file_obj.readlines() # Read the requirements from the file
        requirements= [req.replace("\n","") for req in requirements] # Remove the newline character from the requirements
        
        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)

    return requirements # Return the list of requirements

setup(
name='MLProject',
version='0.0.1',
author='Likhith',
author_email='v.likhith.01@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)