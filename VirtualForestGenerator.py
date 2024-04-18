import subprocess
import sys

blenderExePath      = 'C:\\Users\\luckmanu\\Tools\\Blender\\blender.exe'
blenderFile         = 'D:\\Blender\\VirtualForestGenerator\\Preview\\PreviewScene.blend'
blenderScriptPath   = 'D:\\Blender\\VirtualForestGenerator\\Scripts\\run_console.py'
blenderWorkingDir   = 'C:\\Users\\luckmanu\\Tools\\Blender' 

if len(sys.argv) > 1:
    n = int(sys.argv[1])
else:
    n = 1
    
for _ in range(n):
    subprocess.run([f'{blenderExePath}',f'{blenderFile}','--background',f'--python',f'{blenderScriptPath}'],shell=True,cwd=blenderWorkingDir)

input('Press a key to finish')