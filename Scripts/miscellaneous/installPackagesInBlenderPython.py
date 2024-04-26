import subprocess

blenderPythonPath = C:\\Tools\\Blender\\4.1\\python\\bin\\python.exe

packages = ['numpy','pandas','scipy','fsspec']
for package in packages:
    subprocess.run(f'{blenderPythonPath} -m pip install {package}',
                   shell=True)

input('Finished.\nPress Key to continue')