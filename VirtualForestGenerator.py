import subprocess
import sys

blenderExePath      = 'C:\\Tools\\Blender\\blender.exe'
blenderFile         = 'C:\\Tools\\Python\\Github\\VirtualForestGenerator\\Preview\\PreviewScene.blend'
blenderScriptPath   = 'C:\\Tools\\Python\\Github\\VirtualForestGenerator\\Scripts\\run_console.py'
blenderWorkingDir   = 'C:\\Tools\\Blender'
heliosPath          = 'C:\\Tools\\Python\\Github\\helios\\'

if len(sys.argv) > 1:
    n = int(sys.argv[1])
else:
    n = 1
    
for _ in range(n):
    subprocess.run([f'{blenderExePath}',f'{blenderFile}','--background',f'--python',f'{blenderScriptPath}',f'{heliosPath}'],shell=True,cwd=blenderWorkingDir)

input('Press a key to finish')