# Virtual Forest Generator

The tool aims at generating virtual scans with Helios++ (https://github.com/3dgeo-heidelberg/helios.git) from scenes created in blender (blender.org). 
As default, the resulting point cloud should be labeled according to Point2Tree (https://github.com/SmartForest-no/Point2tree.git) with 4 classes (ground, cwd, vegetation, wood). 

## Installation

(1) Install blender (blender.org)
(2) Clone Helios++ (https://github.com/3dgeo-heidelberg/helios.git)

(3) Clone repo from GitHub (git clone https://github.com/manuelluck/VirtualForestGenerator.git) at the desired location.
(4) In VirtualForestGenerator.py adjust the path according to your blender and Helios++ installation.

(5) Add packages to blenders python. 
    Adjust "/Scripts/miscellaneous/installPackagesInBlenderPython.py" with your blender python path.

## Usage

(5) Run VirtualForestGenerator.py 
	open command console
	optional: "YourRepoDrive:" (e.g., D:)
	cd "your_repo_location" to locate the main folder (cd Data\VirtualForestGenerator)
	python VirtualForestGenerator.py
