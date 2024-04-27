import subprocess

blenderPythonPath = 'C:\\Tools\\Python\\Github\\VirtualForestGenerator\\Blender36\\3.6\\python\\bin\\python.exe'

packages = ['numpy','pandas','scipy','fsspec']
for package in packages:
    subprocess.run(f'{blenderPythonPath} -m pip install {package}',
                   shell=True)

input('Finished.\nPress Key to continue')