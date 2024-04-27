import subprocess
import sys

pathDict = dict()

with open('PathFile.txt') as file:
    for line in file:
        pathDict[line.split(',')[0]] = line.split(',')[1][:-1]

if len(sys.argv) > 1:
    n = int(sys.argv[1])
else:
    n = 1

print('Paths:')
print(pathDict["blenderExePath"])
print(pathDict["blenderFile"])
print(pathDict["blenderScriptPath"])

    
for _ in range(n):
    subprocess.run([f'{pathDict["blenderExePath"]}',f'{pathDict["blenderFile"]}','--background',f'--python',f'{pathDict["blenderScriptPath"]}'],shell=True,cwd=pathDict["blenderWorkingDir"])

input('Press a key to finish')