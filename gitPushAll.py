import subprocess
from pathlib import Path

# Set the path to your local repository
repo_directory = Path(__file__).parent

subprocess.run(['C:\Program Files\Git\cmd\git.exe','add','.'], cwd=repo_directory)
subprocess.run(['C:\Program Files\Git\cmd\git.exe','commit','-m','"autopush"'], cwd=repo_directory)
subprocess.run(['C:\Program Files\Git\cmd\git.exe','push'], cwd=repo_directory)