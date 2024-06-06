# ---------------------------------------------------------------------
#                          Libraries
# ---------------------------------------------------------------------
# System
import os
import sys

# Blender
import bpy
import bmesh

# Arrays & Co
import numpy as np
import pandas as pd
import math

# Kd-Tree
import mathutils

# Filter
from scipy.ndimage import gaussian_filter

# Building environment
import sys

# Console for Helios++
import subprocess

# ---------------------------------------------------------------------
#                    Functions & Parameters
# ---------------------------------------------------------------------
path = dict()
# Main folder
path['main']    = '\\'.join(os.path.dirname(os.path.abspath(__file__)).split('\\')[:-1])

pathDict = dict()
with open(f'{path["main"]}\\PathFile.txt') as f:
    for line in f:
        l = line.rstrip().split(',')
        pathDict[l[0]] = l[1]

# Helios Installation
path['helios'] = pathDict['heliosPath'] if pathDict['heliosPath'].endswith('/') or pathDict['heliosPath'].endswith('\\') else f'{pathDict["heliosPath"]}\\'

# Scripts
path['scripts'] = f'{path["main"]}\\Scripts'

# Assets
path['assets']      = f'{path["main"]}\\Assets'
path['trees']       = f'{path["assets"]}\\TreeParameters'
path['ground_veg']  = f'{path["assets"]}\\GroundVegetation'
path['laying_dw']   = f'{path["assets"]}\\LayingDeadwood'
path['rocks']       = f'{path["assets"]}\\Rocks'
path['tree_stumps'] = f'{path["assets"]}\\TreeStumps'
path['norm_bb']     = f'{path["assets"]}\\NormalizedBuildingBlocks'

# Output
path['output']  = f'{path["main"]}\\Output'

# Preview
path['preview'] = f'{path["main"]}\\Preview'

# Adding to sys to load stuff (blender sometimes does not find files otherwise)
for p in path.values():
    if p not in sys.path:
        sys.path.insert(1,p)

# ---------------------------------------------------------------------
#                          Functions
# ---------------------------------------------------------------------
# Printing
def print4console(text2print, max_l=50):
    print('')
    # split string
    text_list = text2print.split(' ')
    current_l = 0
    split_val = 0
    text_l = [len(text) for text in text_list]
    for i in range(len(text_list)):
        if current_l == 0:
            current_l += text_l[i]
        else:
            current_l += text_l[i] + 1
            
        if current_l >= max_l:
            print(' '.join(text_list[split_val:i]))
            split_val = i
            current_l = 0
        elif i == len(text_list)-1:
            print(' '.join(text_list[split_val:]))

# Importing Objects
def import_obj(filepath, location=(0.0,0.0,0.0),rotation=(0.0,0.0,0.0),scale=(1.0,1.0,1.0),collision=False,permeability=0.3,stickiness=0.25):
    # Import the .obj file
    bpy.ops.import_scene.obj(filepath=filepath)
    
    # Get the imported object
    obj = bpy.context.selected_objects[0]
    
    # Set the object's location
    obj.location        = location
    obj.rotation_euler  = rotation
    obj.scale           = scale
    
    # Adding collision parameters
    if collision:
        obj.modifiers.new(type='COLLISION', name='collision')
        obj.collision.permeability  = permeability
        obj.collision.stickiness    = stickiness
        
    obj.select_set(True)
    
    return obj


def calculate_xy(x, y, l, alpha):
    # Calculate the new x and y coordinates
    new_x = x + l * math.cos(alpha)
    new_y = y + l * math.sin(alpha)

    return new_x, new_y

def calculate_angles(d_x,d_y,d_z):
    # Calculate the lengths of the projections of the slope onto the x, y, and z axes
    length_x = (d_y**2 + d_z**2)**(1/2)
    length_y = (d_x**2 + d_z**2)**(1/2)
    length_z = (d_x**2 + d_y**2)**(1/2)

    # Calculate the angles in radians
    angle_x = np.arccos(d_x / length_x) if length_x != 0 else 0
    angle_y = np.arccos(d_y / length_y) if length_y != 0 else 0
    angle_z = np.arccos(d_z / length_z) if length_z != 0 else 0

    return angle_x, angle_y, angle_z    

def create_directories(structure, root=''):
    for key, value in structure.items():
        if isinstance(value, dict):
            create_directories(value, os.path.join(root, key))
        else:
            os.makedirs(os.path.join(root, value), exist_ok=True)

# ---------------------------------------------------------------------
#                            Class
# ---------------------------------------------------------------------

class worldGenerator():
    # -----------------------------------------------------------------
    # Start up Functions: ---------------------------------------------
    # -----------------------------------------------------------------
    def __init__(self,path,preview_mode=False,save_outputs=True):
        '''
        Inputs: 
            - path: dictionary with all environmental paths
        '''
        # Some initial dictionaries
        # Structural
        self.path       = path
        self.parameters = dict()
        
        # Blender
        self.maps       = dict()
        self.kdTrees    = dict()
        self.vertices   = dict()
        self.objects    = dict()
        
        self.objects['names']   = ['stem','branches','broadleafs','needles']
        
        # Scanpath
        self.scan_path  = []
        
        # Helios
        self.helios                 = dict()
        self.helios['object_files'] = []
        
        # Factor to bypass elevation value in "flat" kd-tree
        self.parameters['kd_factor']    = 1/1000
        
        # Preview saving etc.
        self.parameters['preview_mode'] = preview_mode
        self.parameters['save_outputs'] = save_outputs
        
        if not preview_mode:
            self.parameters['save_outputs'] = True
            
        # Get new scene nr
        if not any(['Output' in dir for dir in os.listdir(path['main'])]):
            os.mkdir(path['output'])

        scenes = os.listdir(path['output'])
        if len(scenes) > 0:
            self.parameters['scene_nr'] = int(max([int(scene.split('scene_nr_')[1]) for scene in scenes]))+1
        else:
            self.parameters['scene_nr'] = 0
           
        # Create folder for the scene
        self.path['scene']= f'{self.path["output"]}\\scene_nr_{str(self.parameters["scene_nr"]).zfill(4)}\\'
        
        if self.parameters['save_outputs']:
            os.mkdir(self.path['scene'])
            
        # Create subfolders for the scene
        self.path['scene_output'] = {'objects':f'{self.path["scene"]}\\objects',
                                     'pointclouds':f'{self.path["scene"]}\\pointclouds',
                                     'vertices':f'{self.path["scene"]}\\vertices',
                                     'helios':f'{self.path["scene"]}\\helios'}

        if self.parameters['save_outputs']:
            create_directories(self.path['scene_output'])        
        
        # Run function to load default tree parameters
        self.get_default_tree_parameters()
        
    def get_default_tree_parameters(self):
        '''
        Loads loads the default tree parameters.
        '''
        
        from White_Birch import parameters as birch
        from Small_Maple import parameters as maple_small
        from Small_Pine import parameters as pine_small  
        from Tall_Pine import parameters as pine_tall
        
        # Store default parameters in parameters dictionary
        self.parameters['default_trees'] = {'pine_small':pine_small,'pine_tall':pine_tall,'birch':birch,'maple_small':maple_small}
        self.deciduous  = ['maple_small','birch']
        self.conifers   = ['pine_small','pine_tall']
        
    
    # -----------------------------------------------------------------    
    # Mathematival Functions: -----------------------------------------
    # -----------------------------------------------------------------
    def generate_rnd_noise_map(self,n_layer,std_dev,n_x,n_y):
        '''
        Generates a random noise scene consisting of multiple layers of noise with different levels of smoothing applied
        Inputs:
            - n_layer   : int   = number of layers 
            - std_dev   : float = starting std value for the smoothing operation
            - n_x       : int   = n-pixels in x dimension
            - n_y       : int   = n-pixels in y dimension
            
        Output:
            - np.Array with shape() = n_x,n_y normalized between [0,1]
            
        '''
        # Generation of matrix
        map = np.zeros((n_x,n_y))
        
        # Looping through the amount of layers n_layers
        for _ in range(n_layer):
            # Generate Normal distributed Noise
            noise = np.random.normal(size=(n_x,n_y))
            
            # Apply smoothing filter
            smooth_noise = gaussian_filter(noise, std_dev)
            
            # Add to map
            map += smooth_noise*(std_dev**2)
            
            # Change standard deviation for the next smoothing
            std_dev *= 3
        
        # Normalize the map and return it
        return (map - np.min(map)) / (np.max(map) - np.min(map))
    
    # -----------------------------------------------------------------
    # Locations and Angles: -------------------------------------------
    # -----------------------------------------------------------------
    def check_possible_targets(self,location,alpha,l):
        targets         = self.kdTrees['kd_3D'].find_range(location,l+0.25)
        alpha_dif_test  = 2*np.pi
        rotation        = (0,0,0)
        
        for target in targets:
            if target[2] > l - 0.25:
                vec = target[0]
                dx,dy,dz = vec[0]-location[0],vec[1]-location[1],vec[2]-location[2]

                if 0 < alpha < np.pi/2:
                    if dx > 0 and dy > 0:                            
                        alpha_target = np.arctan(dy/dx)
                        
                        if abs(alpha-alpha_target) <= np.pi:
                            alpha_dif = abs(alpha-alpha_target)
                        else:
                            alpha_dif = np.pi - abs(alpha-alpha_target)

                        if alpha_dif < alpha_dif_test:
                            epsilon = np.arctan(dz/(dx**2+dy**2)**(1/2)) if dx+dy != 0 else np.pi/2 if dz > 0 else 3*np.pi/2
                            rotation = (0,-epsilon,alpha_target) 
                        
                elif np.pi/2 < alpha < np.pi:
                    if dx < 0 and dy > 0:
                        alpha_target = np.pi+np.arctan(dy/dx)
                        
                        if abs(alpha-alpha_target) <= np.pi:
                            alpha_dif = abs(alpha-alpha_target)
                        else:
                            alpha_dif = np.pi - abs(alpha-alpha_target)

                        if alpha_dif < alpha_dif_test:
                            epsilon = np.arctan(dz/(dx**2+dy**2)**(1/2)) if dx+dy != 0 else np.pi/2 if dz > 0 else 3*np.pi/2
                            rotation = (0,-epsilon,alpha_target) 

                elif np.pi < alpha < 3*np.pi/2:
                    if dx < 0 and dy < 0:                        
                        alpha_target = np.pi+np.arctan(dy/dx)
                        
                        if abs(alpha-alpha_target) <= np.pi:
                            alpha_dif = abs(alpha-alpha_target)
                        else:
                            alpha_dif = np.pi - abs(alpha-alpha_target)

                        if alpha_dif < alpha_dif_test:
                            epsilon = np.arctan(dz/(dx**2+dy**2)**(1/2)) if dx+dy != 0 else np.pi/2 if dz > 0 else 3*np.pi/2
                            rotation = (0,-epsilon,alpha_target) 

                elif 3*np.pi/2 < alpha < 2*np.pi:
                    if dx > 0 and dy < 0:
                        alpha_target = 2*np.pi+np.arctan(dy/dx)
                        
                        if abs(alpha-alpha_target) <= np.pi:
                            alpha_dif = abs(alpha-alpha_target)
                        else:
                            alpha_dif = np.pi - abs(alpha-alpha_target)

                        if alpha_dif < alpha_dif_test:
                            epsilon = np.arctan(dz/(dx**2+dy**2)**(1/2)) if dx+dy != 0 else np.pi/2 if dz > 0 else 3*np.pi/2
                            rotation = (0,-epsilon,alpha_target) 

                elif alpha == 0 or alpha == 2*np.pi:
                    if dx > 0  and dy == 0:
                        alpha_target = 0
                        
                        if abs(alpha-alpha_target) <= np.pi:
                            alpha_dif = abs(alpha-alpha_target)
                        else:
                            alpha_dif = np.pi - abs(alpha-alpha_target)

                        if alpha_dif < alpha_dif_test:
                            epsilon = np.arctan(dz/(dx**2+dy**2)**(1/2)) if dx+dy != 0 else np.pi/2 if dz > 0 else 3*np.pi/2
                            rotation = (0,-epsilon,alpha_target) 

                    elif dx > 0 and 0 < dy < 0.1:
                        alpha_target = np.arctan(dy/dx)
                        
                        if abs(alpha-alpha_target) <= np.pi:
                            alpha_dif = abs(alpha-alpha_target)
                        else:
                            alpha_dif = np.pi - abs(alpha-alpha_target)

                        if alpha_dif < alpha_dif_test:
                            epsilon = np.arctan(dz/(dx**2+dy**2)**(1/2)) if dx+dy != 0 else np.pi/2 if dz > 0 else 3*np.pi/2
                            rotation = (0,-epsilon,alpha_target) 

                    elif dx > 0 and -0.1 < dy < 0:
                        alpha_target = 2*np.pi+np.arctan(dy/dx)
                        
                        if abs(alpha-alpha_target) <= np.pi:
                            alpha_dif = abs(alpha-alpha_target)
                        else:
                            alpha_dif = np.pi - abs(alpha-alpha_target)

                        if alpha_dif < alpha_dif_test:
                            epsilon = np.arctan(dz/(dx**2+dy**2)**(1/2)) if dx+dy != 0 else np.pi/2 if dz > 0 else 3*np.pi/2
                            rotation = (0,-epsilon,alpha_target) 

                elif alpha == np.pi/2:
                    if -0.1 < dx < 0 and dy >= 0:
                        alpha_target = np.pi+np.arctan(dy/dx) if dx != 0 else np.pi/2
                        
                        if abs(alpha-alpha_target) <= np.pi:
                            alpha_dif = abs(alpha-alpha_target)
                        else:
                            alpha_dif = np.pi - abs(alpha-alpha_target)

                        if alpha_dif < alpha_dif_test:
                            epsilon = np.arctan(dz/(dx**2+dy**2)**(1/2)) if dx+dy != 0 else np.pi/2 if dz > 0 else 3*np.pi/2
                            rotation = (0,-epsilon,alpha_target) 

                    elif dx == 0 and dy >= 0:
                        alpha_target = np.pi/2
                        
                        if abs(alpha-alpha_target) <= np.pi:
                            alpha_dif = abs(alpha-alpha_target)
                        else:
                            alpha_dif = np.pi - abs(alpha-alpha_target)

                        if alpha_dif < alpha_dif_test:
                            epsilon = np.arctan(dz/(dx**2+dy**2)**(1/2)) if dx+dy != 0 else np.pi/2 if dz > 0 else 3*np.pi/2
                            rotation = (0,-epsilon,alpha_target) 
                        
                    elif 0 < dx < 0.1 and dy >= 0:
                        alpha_target = np.arctan(dy/dx) if dx != 0 else np.pi/2
                        
                        if abs(alpha-alpha_target) <= np.pi:
                            alpha_dif = abs(alpha-alpha_target)
                        else:
                            alpha_dif = np.pi - abs(alpha-alpha_target)

                        if alpha_dif < alpha_dif_test:
                            epsilon = np.arctan(dz/(dx**2+dy**2)**(1/2)) if dx+dy != 0 else np.pi/2 if dz > 0 else 3*np.pi/2
                            rotation = (0,-epsilon,alpha_target) 

                elif alpha == np.pi:
                    if -0.1 < dy < 0.1 and dx <= 0:
                        alpha_target = np.arctan(dy/dx)
                        
                        if abs(alpha-alpha_target) <= np.pi:
                            alpha_dif = abs(alpha-alpha_target)
                        else:
                            alpha_dif = np.pi - abs(alpha-alpha_target)

                        if alpha_dif < alpha_dif_test:
                            epsilon = np.arctan(dz/(dx**2+dy**2)**(1/2)) if dx+dy != 0 else np.pi/2 if dz > 0 else 3*np.pi/2
                            rotation = (0,-epsilon,alpha_target) 

                elif alpha == 3*np.pi/2:
                    if -0.1 < dx < 0 and dy >= 0:
                        alpha_target = np.pi+np.arctan(dy/dx) if dx != 0 else 3*np.pi/2
                        
                        if abs(alpha-alpha_target) <= np.pi:
                            alpha_dif = abs(alpha-alpha_target)
                        else:
                            alpha_dif = np.pi - abs(alpha-alpha_target)

                        if alpha_dif < alpha_dif_test:
                            epsilon = np.arctan(dz/(dx**2+dy**2)**(1/2)) if dx+dy != 0 else np.pi/2 if dz > 0 else 3*np.pi/2
                            rotation = (0,-epsilon,alpha_target) 

                    elif dx == 0 and dy >= 0:
                        alpha_target = 3*np.pi/2
                        
                        if abs(alpha-alpha_target) <= np.pi:
                            alpha_dif = abs(alpha-alpha_target)
                        else:
                            alpha_dif = np.pi - abs(alpha-alpha_target)

                        if alpha_dif < alpha_dif_test:
                            epsilon = np.arctan(dz/(dx**2+dy**2)**(1/2)) if dx+dy != 0 else np.pi/2 if dz > 0 else 3*np.pi/2
                            rotation = (0,-epsilon,alpha_target) 

                    elif 0 < dx < 0.1 and dy >= 0:
                        alpha_target = 2*np.pi+np.arctan(dy/dx) if dx != 0 else 3*np.pi/2
                        
                        if abs(alpha-alpha_target) <= np.pi:
                            alpha_dif = abs(alpha-alpha_target)
                        else:
                            alpha_dif = np.pi - abs(alpha-alpha_target)

                        if alpha_dif < alpha_dif_test:
                            epsilon = np.arctan(dz/(dx**2+dy**2)**(1/2)) if dx+dy != 0 else np.pi/2 if dz > 0 else 3*np.pi/2
                            rotation = (0,-epsilon,alpha_target) 
                else:
                    print('Point not in range!')   
                        
        return rotation 
        
    def get_stretch_locations(self,location,radius,lower_factor=1/3,elevate_factor=1/2):
        targets         = self.kdTrees['kd_3D'].find_range(location,radius)
        location_max    = (0,0,-9999)
        z_min           = 9999
        for target in targets:
            if target[0][2] > location_max[2]:
                location_max = (target[0][0],target[0][1],target[0][2])
            if target[0][2] < z_min:
                z_min       = target[0][2]
        dz = location_max[2]-z_min
        
        location_empty  = (location_max[0],location_max[1],location_max[2]+dz*elevate_factor)
        location        = (location[0],location[1],location[2]-dz*lower_factor)
        return location, location_empty               
               
    
    def calculate_z(self,pos,lower=0,modus='fit_plane'):
        pos = [pos[0],pos[1],0]
        
        co = np.zeros([3,3])
        for i,(coord,_,_) in enumerate(self.kdTrees['kd_flat'].find_n(pos,3)):
            co[i,:] = coord
            co[i,2] /= self.parameters['kd_factor']
        if modus == 'fit_plane':    
            # calculate vectors AB and AC
            AB = co[1,:] - co[0,:]
            AC = co[2,:] - co[0,:]

            # calculate the normal vector by cross product
            normal = np.cross(AB, AC)
            
            # calculate D
            D = -np.dot(normal, co[0,:])

            # calculate z
            pos[2] = (-D - normal[0]*pos[0] - normal[1]*pos[1]) / normal[2]
            
            if pos[2] == -np.inf or pos[2] == np.inf:
                pos[2] = co[0,2]+lower
        elif modus == 'knn':
            co = np.nanmean(co,axis=0)
            pos[2] = co[2]+lower
        
        return pos    
    
    def get_vertices_for_obj(self, obj):
        
        # Get latest version of the object
        obj.data.update()

        # Get the object vertices
        vertices = obj.data.vertices

        # Get DEM vertices
        idx_list = []
        for vertex in vertices:
            _,idx,_ = self.kdTrees['kd_3D'].find(obj.matrix_world @ vertex.co)
            idx_list.append(idx)
        
        # return unique values
        return np.unique(idx_list)
    
    def export_vertices(self):
        if self.parameters['save_outputs']:
            with open(f'{self.path["scene_output"]["vertices"]}\\vertices_dict.py','w') as f:
                f.write(f'vertices = {self.vertices}')
            
    def import_vertices(self):
        sys.path.insert(1,self.path['scene_output']['vertices'])
        
        from vertices_dict import vertices   
        
        # Store default parameters in parameters dictionary
        self.vertices = vertices        
            
    # -----------------------------------------------------------------
    # Maps and Vertices Info: -----------------------------------------
    # -----------------------------------------------------------------    
    
    # Creating Random Digital Elevation Models
    def add_DEM(self,DEM=None,create_rnd=True,n_layer=5,std_dev=5.0,n_x=200,n_y=200,d_x=0.1,d_y=0.1,d_z=1,boundry=0):
        
        self.parameters['noise_layers']     = n_layer
        self.parameters['start_std_noise']  = std_dev
        
        self.parameters['n_x'] = n_x
        self.parameters['n_y'] = n_y
        self.parameters['d_x'] = d_x
        self.parameters['d_y'] = d_y
        self.parameters['d_z'] = d_z
        
        self.parameters['extend_x'] = n_x*d_x
        self.parameters['extend_y'] = n_y*d_y  
        
        self.parameters['boundry']  = boundry     
        
        if DEM == None and create_rnd:
            # Create random noise map
            self.maps['DEM'] = self.generate_rnd_noise_map(n_layer=n_layer,std_dev=std_dev,n_x=n_x,n_y=n_y)*d_z
        else:
            self.maps['DEM'] = DEM
        
        # Spawn a new grid with the desired dimensions
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=n_x-1, y_subdivisions=n_y-1, size=1, enter_editmode=True, align='WORLD', location=(0, 0, 0), scale=(n_x*d_x, n_y*d_y, 1))
        
        # Select the new grid
        DEM = bpy.context.object

        # Change to Object-Mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # Initalize i and j as counters for the x,y dimension of the grid
        i = 0
        j = 0
        
        # Initialize kd-tree
        self.kdTrees['kd_flat'] = mathutils.kdtree.KDTree(len(DEM.data.vertices))
        self.kdTrees['kd_3D']   = mathutils.kdtree.KDTree(len(DEM.data.vertices))
        
        # Loop through all the vertices of the mesh
        for idx,v in enumerate(DEM.data.vertices):
            if i < n_x: # If within one line in the grid
                # Adjust the location of each vertex
                v.co.z = self.maps['DEM'][i,j]  # Values from the peviously generated DEM scaled by factor d_z
                v.co.x = i*d_x - ((n_x/2)*d_x)  # Grid position scaled by factor d_x and centered around (0,0)
                v.co.y = j*d_y - ((n_y/2)*d_y)  # Grid position scaled by factor d_y and centered around (0,0)

                # Add to kd-tree
                self.kdTrees['kd_flat'].insert([v.co.x,v.co.y,v.co.z*self.parameters['kd_factor']], idx)
                self.kdTrees['kd_3D'].insert([v.co.x,v.co.y,v.co.z],idx)
                if abs(v.co.x) < self.parameters['extend_x']/2-self.parameters['boundry'] or abs(v.co.y) < self.parameters['extend_y']/2-self.parameters['boundry']:
                    self.vertices[idx] = {'idx':idx,
                                          'img_coord':[i,j],
                                          'blend_coord':[v.co.x,v.co.y,v.co.z],
                                          'tree':False,
                                          'understory_tree':False,
                                          'dw':False,
                                          'low_veg':False,
                                          'rock':False,
                                          'occupied':False,
                                          'accessible':True,
                                          'closed_crown':False
                                          }
                else:
                    self.vertices[idx] = {'idx':idx,
                                          'img_coord':[i,j],
                                          'blend_coord':[v.co.x,v.co.y,v.co.z],
                                          'tree':False,
                                          'understory_tree':False,
                                          'dw':False,
                                          'low_veg':False,
                                          'rock':False,
                                          'occupied':True,
                                          'accessible':True,
                                          'closed_crown':False
                                          }                      
                
                # Next one in line
                i += 1
                
            else: # If end of the line
                # Switching line
                j += 1
                i = 0
                
                # Adjust the location of each vertex
                v.co.z = self.maps['DEM'][i,j]  # Values from the peviously generated DEM scaled by factor d_z
                v.co.x = i*d_x - ((n_x/2)*d_x)  # Grid position scaled by factor d_x and centered around (0,0)
                v.co.y = j*d_y - ((n_y/2)*d_y)  # Grid position scaled by factor d_y and centered around (0,0)

                # Add to kd-tree
                self.kdTrees['kd_flat'].insert([v.co.x,v.co.y,v.co.z*self.parameters['kd_factor']], idx)
                self.kdTrees['kd_3D'].insert([v.co.x,v.co.y,v.co.z],idx)
                self.vertices[idx] = {'idx':idx,
                                      'img_coord':[i,j],
                                      'blend_coord':[v.co.x,v.co.y,v.co.z],
                                      'tree':False,
                                      'understory_tree':False,
                                      'dw':False,
                                      'low_veg':False,
                                      'rock':False,
                                      'occupied':False,
                                      'accessible':True,
                                      'closed_crown':False
                                      }                
                # Next one in line
                i += 1              

        # Update the mesh
        DEM.data.update()
        
        # Balance kd-tree
        self.kdTrees['kd_flat'].balance()
        self.kdTrees['kd_3D'].balance()
        
        # Rename Object
        DEM.name = f'DEM'
        # Store DEM Object    
        self.objects['DEM'] = DEM  
        
        file_path = self.path['scene_output']['objects']+f'\\ground.obj'
        self.helios['object_files'].append(file_path)
        

        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = DEM
        DEM.select_set(True)
        
        if self.parameters['save_outputs']:
            bpy.ops.export_scene.obj(
                        filepath=file_path,
                        use_selection=True,
                        axis_up='Z',
                    )
                 
                
    def add_CHM(self,CHM=None,create_rnd=True,n_layer=5,std_dev=5.0,max_height=35):
        if CHM != None:
            self.maps['CHM'] = CHM
        else:
            self.maps['CHM'] = self.generate_rnd_noise_map(n_layer=n_layer,
                                                           std_dev=std_dev,
                                                           n_x=self.parameters['n_x'],
                                                           n_y=self.parameters['n_y']
                                                           )*max_height
        
        self.parameters['max_CHM_height'] = max_height
            
        if np.shape(self.maps['DEM']) == np.shape(self.maps['CHM']):
            i = 0
            j = 0
            for idx in range(len(self.vertices)):    
                if i < len(self.maps['CHM'][0,:]): # If within one line in the grid
                    # Add data
                    self.vertices[idx]['canopy_height'] = self.maps['CHM'][i,j]
                    
                    # Next in line
                    i += 1
    
                else: # If end of the line
                    # Switching line
                    j += 1
                    i = 0
                    
                    # Add data
                    self.vertices[idx]['canopy_height'] = self.maps['CHM'][i,j]
           
                    # Next one in line
                    i += 1
                    
    def add_ground_vegetation_map(self,map=None,covarage_class=4,create_rnd=True,n_layer=5,std_dev=5.0):
        if map != None:
            self.maps['ground_veg'] = map
        else:    
            map = self.generate_rnd_noise_map(n_layer=n_layer,
                                              std_dev=std_dev,
                                              n_x=self.parameters['n_x'],
                                              n_y=self.parameters['n_y']
                                              )
            
            if covarage_class == 1:
                threshold = np.percentile(map[:],1)
            elif covarage_class == 2:
                threshold = np.percentile(map[:],5)
            elif covarage_class == 3:
                threshold = np.percentile(map[:],17.5)
            elif covarage_class == 4:
                threshold = np.percentile(map[:],37.5)
            elif covarage_class == 5:
                threshold = np.percentile(map[:],62.5)
            elif covarage_class == 6:
                threshold = np.percentile(map[:],87.5)
            
            self.maps['ground_veg'] = map < threshold 
         
        if np.shape(self.maps['DEM']) == np.shape(self.maps['ground_veg']):
            i = 0
            j = 0
            for idx in range(len(self.vertices)):    
                if i < len(self.maps['ground_veg'][0,:]): # If within one line in the grid
                    # Add data
                    self.vertices[idx]['ground_veg'] = self.maps['ground_veg'][i,j]
                    
                    # Next in line
                    i += 1
    
                else: # If end of the line
                    # Switching line
                    j += 1
                    i = 0
                    
                    # Add data
                    self.vertices[idx]['ground_veg'] = self.maps['ground_veg'][i,j]
           
                    # Next one in line
                    i += 1            
        
                    
    def add_dominante_leaf_type(self,leaf_type_map=None,create_rnd=True,type_threshold=0.5,n_layer=5,std_dev=5.0,sigma=0.1):
        '''
        1 -> deciduous
        0 -> conifer
        0 < x < 1 mixed forest (if 80% conifer 0 - 0.8 conifers)
        '''
        if leaf_type_map != None:
            self.maps['dominant_leaf_type'] = leaf_type_map
        else:
            self.maps['dominant_leaf_type'] = self.generate_rnd_noise_map(n_layer=n_layer,
                                                                          std_dev=std_dev,
                                                                          n_x=self.parameters['n_x'],
                                                                          n_y=self.parameters['n_y']
                                                                          )
            
        if np.shape(self.maps['DEM']) == np.shape(self.maps['dominant_leaf_type']):
            i = 0
            j = 0
            for idx in range(len(self.vertices)):    
                if i < len(self.maps['dominant_leaf_type'][0,:]): # If within one line in the grid
                    # Add data
                    self.vertices[idx]['dominant_leaf_type']   = self.maps['dominant_leaf_type'][i,j]
                    if leaf_type_map == None:
                        self.vertices[idx]['tree_type']     = 'conifer' if self.vertices[idx]['dominant_leaf_type'] <= type_threshold else 'deciduous'
                    else:
                        self.vertices[idx]['tree_type']     = 'conifer' if np.random.normal(self.vertices[idx]['dominant_leaf_type'], sigma, n) < type_threshold else 'deciduous'                     
                        
                    # Next in line
                    i += 1
    
                else: # If end of the line
                    # Switching line
                    j += 1
                    i = 0
                    
                    # Add data
                    self.vertices[idx]['dominant_leaf_type']   = self.maps['dominant_leaf_type'][i,j]
                    if leaf_type_map == None:
                        self.vertices[idx]['tree_type']     = 'conifer' if self.vertices[idx]['dominant_leaf_type'] <= type_threshold else 'deciduous'
                    else:
                        self.vertices[idx]['tree_type']     = 'conifer' if np.random.normal(self.vertices[idx]['dominant_leaf_type'], sigma, n) < type_threshold else 'deciduous' 
           
                    # Next one in line
                    i += 1     

    # -----------------------------------------------------------------
    # Spawn Objects: --------------------------------------------------
    # -----------------------------------------------------------------  

    # Planting broad leafed plant
    def spawnBroadLeafedPlant(self,objpath,n_sets=10,set_size=25,scaling=1.5,remove_after_save=False):
        
        obs     = []
        rnd_arr = np.random.rand(n_sets,2)
        
        for x,y in rnd_arr:
            x = (x*self.parameters['extend_x']) - (self.parameters['extend_x']/2)
            y = (y*self.parameters['extend_y']) - (self.parameters['extend_y']/2)
            # bpy.ops.mesh.primitive_cylinder_add(vertices=3, radius=0.01, depth=5, enter_editmode=False, align='WORLD', location=(x, y, 0), scale=(1, 1, 1))
            for _ in range(set_size):
                [shift_x,shift_y] = np.random.normal(0, 0.1, 2)
                
                location            = self.calculate_z([x+shift_x,y+shift_y])
                if shift_x >= 0:
                    rotation = (0,0,np.arctan(shift_y/shift_x)+np.radians(90))
                else:
                    rotation = (0,0,np.arctan(shift_y/shift_x)+np.radians(270))
                    
                d = (shift_x**2+shift_y**2)**(1/2)
                
                s = (1-np.arctan(d*10)/(np.pi/2))*scaling   
                scale    = (s,s,s)
                
                obj = import_obj(objpath, location=location, rotation=rotation, scale=scale)
                obs.append(obj)    
            
        bpy.ops.object.select_all(action='DESELECT')
        for ob in obs:
            if ob.type == 'MESH':
                ob.select_set(True)
                bpy.context.view_layer.objects.active = ob
            ob.name = 'broad_leaf_plant'
        
        self.objects['names'].append('broad_leaf_plant')

    def spawnDeadWoodStemSamples(self,obj_folder,n_stems=10,remove_after_save=False):
        obs = []
        df = pd.DataFrame(columns=['id', 'path', 'diameter', 'length'])
        counter = 0
        files = os.listdir(obj_folder)
        for file in files:
            if file.endswith('.obj'):
                filepath = os.path.join(obj_folder,file)
                
                parts = file.split('_')
                for part in parts:
                    if part.startswith('d'):
                        try:
                            d = int(part[1:4])
                        except:
                            d = -1
                    elif part.startswith('l'):
                        try:
                            l = int(part[1:5])
                        except:
                            l = -1
                        
                # Append a new row to the DataFrame
                df.loc[len(df.index)] = [counter, filepath, d, l]
                counter +=1
        for _ in range(n_stems):   
            # Choose stem obj
            i       = np.random.randint(0,len(df['length']))
            stem    = df.iloc[i]
            
            # Get Diameter and length in Blender units
            l   = stem.loc['length']/100
            d   = stem.loc['diameter']/100            
               
            # Random position
            [[x,y]]   = np.random.rand(1,2)
            x = (x*(self.parameters['extend_x']-l)) - ((self.parameters['extend_x']-l)/2)
            y = (y*(self.parameters['extend_y']-l)) - ((self.parameters['extend_y']-l)/2)
            
            # Find DEM elevation
            location    = self.calculate_z([x,y]) 
            
            # Random Rotation alonge z
            alpha = np.random.uniform(0,np.pi*2,1)            
            
            # Get rotation for endpoints and midpoint on kd-tree based on distance and alpha
            rotation_end = self.check_possible_targets(location,alpha,l)
            rotation_mid = self.check_possible_targets(location,alpha,l/2)
            
            rotation = rotation_end if rotation_end[1] < rotation_mid[1] else rotation_mid
            
            # Elevate stem by 1/3 of diameter
            location[2] += d/3
            
            # Import the object with the given location and rotation
            obj = import_obj(stem.loc['path'], location=location, rotation=rotation, scale=(1,1,1))
            
            # Add obj to list ob objects
            obj.data.update()
            obs.append(obj) 
                                       
        # Join objects    
        bpy.ops.object.select_all(action='DESELECT')
        for ob in obs:
            if ob.type == 'MESH':
                ob.select_set(True)
                bpy.context.view_layer.objects.active = ob
                ob.name = 'laying_deadwood'
                
                idx_list = self.get_vertices_for_obj(ob)
                for idx in idx_list:
                    self.vertices[idx]['occupied']  = True
                    self.vertices[idx]['dw']        = True
                    self.vertices[idx]['obj_type']  = 'laying_dw'  
                    
        self.objects['names'].append('laying_deadwood')
                                  
    def spawnRocks(self,obj_folder,n_rocks=50,remove_after_save=False):
        obs = []
        df = pd.DataFrame(columns=['id', 'path'])
        counter = 0
        files = os.listdir(obj_folder)
        for file in files:
            if file.endswith('.obj'):
                filepath = os.path.join(obj_folder,file)
                    
                # Append a new row to the DataFrame
                df.loc[len(df.index)] = [counter, filepath]
                counter +=1

        for _ in range(n_rocks):   
            # Choose stem obj
            i       = np.random.randint(0,len(df['id']))
            rock    = df.iloc[i]  
                          
            # Random Rotation alonge z
            alpha = np.random.uniform(0,np.pi*2,1)
            beta  = alpha+np.pi/2 if alpha < 3*np.pi/2 else alpha-np.pi/2
                     
            # Random position
            [[x,y]]   = np.random.rand(1,2)
            x = (x*(self.parameters['extend_x']-2)) - ((self.parameters['extend_x']-2)/2)
            y = (y*(self.parameters['extend_y']-2)) - ((self.parameters['extend_y']-2)/2)
            
            # Find DEM elevation
            location            = self.calculate_z([x,y])
            rotation            = self.check_possible_targets(location,alpha,0.5)
            _,a,_               = self.check_possible_targets(location,beta,0.5)
            rotation            = (a,rotation[1],rotation[2])
            
            # Import the object with the given location and rotation
            obj = import_obj(rock.loc['path'], location=location, rotation=rotation, scale=(1,1,1))
            
            # Add obj to list ob objects
            obj.data.update()
            obs.append(obj) 
                       
        # Join objects    
        bpy.ops.object.select_all(action='DESELECT')
        for ob in obs:
            if ob.type == 'MESH':
                ob.select_set(True)
                bpy.context.view_layer.objects.active = ob
                ob.name = 'rock'
                
                idx_list = self.get_vertices_for_obj(ob)
                for idx in idx_list:
                    self.vertices[idx]['occupied']  = True
                    self.vertices[idx]['rock']      = True
                    self.vertices[idx]['obj_type']  = 'rock'  
                    
        self.objects['names'].append('rock')       
            
    def spawnStumps(self,obj_folder,n_stumps=2,radius=1,lower_factor=1/3,elevate_factor=1/2,remove_after_save=False):
        obs = []
        df = pd.DataFrame(columns=['id', 'path'])
        counter = 0
        files = os.listdir(obj_folder)
        for file in files:
            if file.endswith('.obj'):
                filepath = os.path.join(obj_folder,file)
                    
                # Append a new row to the DataFrame
                df.loc[len(df.index)] = [counter, filepath]
                counter +=1

        for _ in range(n_stumps):   
            # Choose stem obj
            i       = np.random.randint(0,len(df['id']))
            stump   = df.iloc[i]  
                     
            # Random position
            [[x,y]]   = np.random.rand(1,2)
            x = (x*self.parameters['extend_x']) - (self.parameters['extend_x']/2)
            y = (y*self.parameters['extend_y']) - (self.parameters['extend_y']/2)
            
            # Find DEM elevation
            location                    = self.calculate_z([x,y])            
            location,location_empty     = self.get_stretch_locations(np.array(location)-[0,0,0.2],radius=radius,
                                                                     lower_factor=lower_factor,
                                                                     elevate_factor=elevate_factor)
            
            # 
            s = np.random.uniform(0.3,1.3,1)
            
            # Import the object with the given location and rotation
            obj = import_obj(stump.loc['path'], location=location, rotation=(0,0,np.random.uniform(-np.pi,np.pi,1)), scale=(s,s,s))
            
            bpy.ops.object.empty_add(type='SPHERE', align='WORLD', location=location_empty, scale=(1, 1, 1))
            empty = bpy.context.object
            
            # Set modifier
            obj.modifiers.new("SimpleDeform_x", 'SIMPLE_DEFORM')

            # Set the modifier properties
            modifier_x = obj.modifiers["SimpleDeform_x"]
            modifier_x.deform_method = 'STRETCH'
            modifier_x.origin = empty
            modifier_x.deform_axis = 'X'
            modifier_x.factor = 0.5
            modifier_x.lock_y = True
                        
            # Set modifier
            obj.modifiers.new("SimpleDeform_y", 'SIMPLE_DEFORM')

            # Set the modifier properties
            modifier_y = obj.modifiers["SimpleDeform_y"]
            modifier_y.deform_method = 'STRETCH'
            modifier_y.origin = empty
            modifier_y.deform_axis = 'Y'
            modifier_y.factor = 0.5
            modifier_y.lock_x = True
            
            # Apply modifier
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply(modifier="SimpleDeform_x")
            bpy.ops.object.modifier_apply(modifier="SimpleDeform_y")
            
            # Remove empty
            bpy.data.objects.remove(empty, do_unlink=True)      
                     
            # Add obj to list ob objects
            obj.data.update()
            obs.append(obj)
            idx_list = self.get_vertices_for_obj(obj)
            for idx in idx_list:
                self.vertices[idx]['occupied']  = True
                self.vertices[idx]['dw']        = True
                self.vertices[idx]['obj_type']  = 'stump'             
                       
        # Join objects    
        bpy.ops.object.select_all(action='DESELECT')
        for ob in obs:
            if ob.type == 'MESH':
                ob.select_set(True)
                bpy.context.view_layer.objects.active = ob
                ob.name = 'stump'
        self.objects['names'].append('stump')
        
    def seed_rnd_trees(self,n_trees):
        # get uniform rnd nr of vertices positions to plant tree
        idx_list        = np.random.randint(len(self.vertices),size=n_trees)
        self.tree_seeds = []
        
        for idx in idx_list:
            counter = 0
            while any([self.vertices[n_idx]['occupied'] for (_,n_idx,_) in self.kdTrees['kd_3D'].find_range(self.vertices[idx]['blend_coord'],radius=self.vertices[idx]['canopy_height']*0.03)]) and counter <= 100:
                idx = np.random.randint(len(self.vertices),size=1)[0]
                counter += 0
            if counter <= 100:    
                self.vertices[idx]['tree']      = True
                self.vertices[idx]['occupied']  = False
                
                
                for (_,n_idx,_) in self.kdTrees['kd_3D'].find_range(self.vertices[idx]['blend_coord'],radius=self.vertices[idx]['canopy_height']*0.03):
                    self.vertices[n_idx]['occupied']    = True
                    self.vertices[n_idx]['accessible']  = False
                    self.vertices[n_idx]['obj_type']    = 'tree'
                    
                self.tree_seeds.append(self.vertices[idx])
            
    def get_neighbour_information(self,height_distance_factor=3):
        for seed in self.tree_seeds:
            neighbour_counter   = 0
            neighbour_d         = []
            neighbour_h         = []
            for i,(_,idx,d) in enumerate(self.kdTrees['kd_3D'].find_range(seed['blend_coord'],radius=seed['canopy_height']/height_distance_factor)):
                if self.vertices[idx]['tree']:
                    if 0.8 < self.vertices[idx]['canopy_height']/seed['canopy_height'] < 1.2:
                        neighbour_counter += 1
                        neighbour_d.append(d)
                        neighbour_h.append(self.vertices[idx]['canopy_height'])               
                    
            seed['mean_neighbour_d'] = np.mean(neighbour_d)
            seed['mean_neighbour_h'] = np.mean(neighbour_h)
            seed['neighbour_count']  = neighbour_counter

            if neighbour_counter >= 3:
                seed['closed_crown']    = True  
            else:
                seed['closed_crown']    = False
                
    def grow_trees(self,showLeaves=True,vFactor=0.1,decimate=0.0872665,join_species=True,remove_after_save=False):
        def r(n, r0=0.2, rinf=0.1, k=0.5):
            if rinf < r0:
                return rinf+((r0-rinf)*np.exp(-k*n))
            else:
                return rinf-((rinf-r0)*np.exp(-k*n))        
            
        # Running through all seed vertices
        for seed in self.tree_seeds:
            # Checking for the dominate leaf type
            if seed['tree_type'] == 'deciduous':
                seed['species'] = np.random.choice(self.deciduous)
                
                # copy the default parameters
                seed['tree_parameters']  = self.parameters['default_trees'][seed['species']].copy()
            else:
                if seed['canopy_height'] > 8:
                    seed['species'] = 'pine_tall'
                
                    # copy the default parameters
                    seed['tree_parameters']  = self.parameters['default_trees']['pine_tall'].copy()                     
                
                else:
                    seed['species'] = 'pine_small'
                
                    # copy the default parameters
                    seed['tree_parameters']  = self.parameters['default_trees']['pine_small'].copy() 
            
            
            # Store default parameterset:
            default_parameters = seed.copy()
                    
            # Adjust parameters based on seed values
            # CHM
            seed['tree_parameters']['scale']   = seed['canopy_height'] if seed['canopy_height'] > 1.5 else 1.5
            
            # Get the scale ration between default scale and CHM scale
            scale_ratio = seed['tree_parameters']['scale']/default_parameters['tree_parameters']['scale']
            
            # Variation in CHM
            seed['tree_parameters']['scaleV']  = seed['tree_parameters']['scale']*vFactor
            
            # Amount of subbranches based on size
            seed['tree_parameters']['levels']  = 5 if seed['tree_parameters']['scale'] > 35 else 4 if seed['tree_parameters']['scale'] > 20 else 3 if seed['tree_parameters']['scale'] > 10 else 2
            
            # Total Amount of branch rings
            seed['tree_parameters']['nrings'] = int(seed['tree_parameters']['scale']*2)
    
            # Amount of branches
            branches = [int(i*scale_ratio) for i in seed['tree_parameters']['branches']]
            seed['tree_parameters']['branches'] = (branches[0],branches[1],branches[2],branches[3])
            
            # Size of the branches
            seed['tree_parameters']['ratioPower'] = 1.1 if seed['tree_parameters']['scale'] > 30 else 1.066 if seed['tree_parameters']['scale'] > 20 else 1.033 if seed['tree_parameters']['scale'] > 10 else 1
            
            # Calcualte DBH
            seed['DBH'] = (seed['tree_parameters']['scale']*seed['tree_parameters']['ratio']*seed['tree_parameters']['scale0'])*2
            
            # Rnd Seed
            seed['tree_parameters']['seed'] = np.random.randint(1000)
            
            # Splitting
            if seed['tree_parameters']['baseSplits'] > 0:
                seed['tree_parameters']['baseSplits'] = np.random.randint(seed['tree_parameters']['baseSplits']-1,seed['tree_parameters']['baseSplits']+2)  
                
            shape = seed['tree_parameters']['customShape'] 
            # Close Crown
            if seed['closed_crown']:
                if seed['species'] == 'pine_small':
                    seed['tree_parameters']['shape']            = '8'
                    seed['tree_parameters']['customShape']      = (0.2,0.2,0.8,0.9)
                
                elif seed['species'] == 'white_birch':
                    seed['tree_parameters']['shape']            = '8'
                    seed['tree_parameters']['customShape']      = (0.2,0.4,0.8,0.9)
                    
                    splitAngle                                  = seed['tree_parameters']['splitAngle']
                    seed['tree_parameters']['splitAngle']       = (splitAngle[0]/1.2,splitAngle[1],splitAngle[2],splitAngle[3])
                    
                    splitAngleV                                 = seed['tree_parameters']['splitAngleV']
                    seed['tree_parameters']['splitAngleV']      = (splitAngleV[0]/1.2,splitAngleV[1],splitAngleV[2],splitAngleV[3])
                elif seed['species'] == 'maple_small':
                    seed['tree_parameters']['shape']            = '8'
                    seed['tree_parameters']['customShape']      = (r(seed['neighbour_count'],shape[0],shape[0]/2),
                                                                   r(seed['neighbour_count'],shape[1],shape[1]/3),
                                                                   r(seed['neighbour_count'],shape[2],0.9),
                                                                   r(seed['neighbour_count'],shape[3],shape[3]/4))
                    
            # Adjust parameter based on manual input
            # Show leaves
            seed['tree_parameters']['showLeaves'] = showLeaves
                        
            bpy.ops.curve.tree_add(**seed['tree_parameters'])
            
            ## Leaves
            if seed['tree_parameters']['showLeaves']:
                # Deselect all points in the curve
                bpy.ops.object.select_all(action='DESELECT')

                # Select in data
                leaves = bpy.data.objects['leaves']

                if seed['tree_parameters']['leafShape'] != 'rect':
                    leaves.name = 'broadleafs'
                else:
                    leaves.name = 'needles'

                # Select the curve object and make it the active object
                leaves.select_set(True)
                bpy.context.view_layer.objects.active = leaves

                # Clear Parent
                bpy.ops.object.parent_clear(type='CLEAR')
                
                leaves.location   = seed['blend_coord'] 
                
            ## Stem & Branches & Twigs
            # Deselect all points in the curve
            bpy.ops.object.select_all(action='DESELECT')

            # Select in data
            tree = bpy.data.objects['tree']

            tree.name       = 'temporary'
            tree.location   = seed['blend_coord'] 

            # Select the curve object and make it the active object
            tree.select_set(True)
            bpy.context.view_layer.objects.active = tree

            # Go into edit mode
            bpy.ops.object.mode_set(mode='EDIT')

            # Deselect all points in the curve
            bpy.ops.curve.select_all(action='DESELECT')
            for stem in range(seed['tree_parameters']['baseSplits']+1):
                for point in tree.data.splines[stem].bezier_points:
                    point.select_control_point = True

            # Separate the selected points into a new object
            bpy.ops.curve.separate()

            # Go back to object mode
            bpy.ops.object.mode_set(mode='OBJECT')

            bpy.data.objects['temporary'].name      = 'branches'
            bpy.data.objects['temporary.001'].name  = 'stem'
            
    def seed_understory_trees(self,distance=5,n_trees=3):
        self.understory_seeds = []
        for seed in self.tree_seeds:
            idx_list = []
            for (_,idx,_) in self.kdTrees['kd_3D'].find_range(seed['blend_coord'],radius=distance):
                if self.vertices[idx]['canopy_height'] > 4 and not any([self.vertices[n_idx]['occupied'] for (_,n_idx,_) in self.kdTrees['kd_3D'].find_range(self.vertices[idx]['blend_coord'],radius=0.2)]):
                    idx_list.append(idx)
                    
            if len(idx_list) > 0:        
                rnd_idx = np.random.randint(len(idx_list),size=n_trees)
                for i in rnd_idx:
                    counter = 0 
                    while any([self.vertices[n_idx]['occupied'] for (_,n_idx,_) in self.kdTrees['kd_3D'].find_range(self.vertices[idx_list[i]]['blend_coord'],radius=0.2)]) and counter <= 500:
                         i = np.random.randint(len(idx_list),size=1)[0]
                         counter += 1
                    if counter <= 500:
                        self.vertices[idx_list[i]]['understory_tree']   = True
                        self.vertices[idx_list[i]]['occupied']          = True
                        
                        for (_,n_idx,_) in self.kdTrees['kd_3D'].find_range(self.vertices[idx_list[i]]['blend_coord'],radius=0.25):
                            self.vertices[n_idx]['occupied']        = True
                            self.vertices[n_idx]['accessible']      = False
                            self.vertices[n_idx]['understory_tree'] = True
                            self.vertices[n_idx]['obj_type']        = 'understory_tree'                 
                        
                        self.understory_seeds.append(self.vertices[idx_list[i]])
                          
    def grow_understory_trees(self,showLeaves=True,vFactor=0.1,decimate=0.0872665,join_species=True,remove_after_save=False):        
        # Running through all seed vertices
        for seed in self.understory_seeds:
            # Checking for the dominate leaf type
            if seed['tree_type'] == 'deciduous':
                seed['species']     = 'white_birch'
                # copy the default parameters
                seed['tree_parameters']  = self.parameters['default_trees']['birch'].copy()
            else:
                seed['species']     = 'pine_small'
                
                # copy the default parameters
                seed['tree_parameters']  = self.parameters['default_trees']['pine_small'].copy() 
            
            
            # Store default parameterset:
            default_parameters = seed.copy()
                    
            # Adjust parameters based on seed values
            seed['tree_parameters']['scale']   = 4
            seed['tree_parameters']['scaleV']  = 2
            
            # Get the scale ration between default scale and CHM scale
            scale_ratio = seed['tree_parameters']['scale']/default_parameters['tree_parameters']['scale']
            
            # Amount of subbranches based on size
            seed['tree_parameters']['levels']  =  2
            
            # Total Amount of branch rings
            seed['tree_parameters']['nrings'] = 10
    
            # Amount of branches
            branches = [int(i*scale_ratio) for i in seed['tree_parameters']['branches']]
            seed['tree_parameters']['branches'] = (branches[0],branches[1],branches[2],branches[3])
            
            # Size of the branches
            seed['tree_parameters']['ratioPower'] = 1
            
            # Calcualte DBH
            seed['DBH'] = (seed['tree_parameters']['scale']*seed['tree_parameters']['ratio']*seed['tree_parameters']['scale0'])*2
            
            # Rnd Seed
            seed['tree_parameters']['seed'] = np.random.randint(1000)
            
            # Splitting
            if seed['tree_parameters']['baseSplits'] > 0:
                seed['tree_parameters']['baseSplits'] = np.random.randint(seed['tree_parameters']['baseSplits']-1,seed['tree_parameters']['baseSplits']+1)  
                    
            # Adjust parameter based on manual input
            # Show leaves
            seed['tree_parameters']['showLeaves'] = showLeaves
                        
            bpy.ops.curve.tree_add(**seed['tree_parameters'])
            
            ## Leaves
            if seed['tree_parameters']['showLeaves']:
                # Deselect all points in the curve
                bpy.ops.object.select_all(action='DESELECT')

                # Select in data
                leaves = bpy.data.objects['leaves']

                if seed['tree_parameters']['leafShape'] != 'rect':
                    leaves.name = 'broadleafs'
                else:
                    leaves.name = 'needles'

                # Select the curve object and make it the active object
                leaves.select_set(True)
                bpy.context.view_layer.objects.active = leaves

                # Clear Parent
                bpy.ops.object.parent_clear(type='CLEAR')
                
                leaves.location   = seed['blend_coord'] 
                
            ## Stem & Branches & Twigs
            # Deselect all points in the curve
            bpy.ops.object.select_all(action='DESELECT')

            # Select in data
            tree = bpy.data.objects['tree']

            tree.name       = 'temporary'
            tree.location   = seed['blend_coord'] 

            # Select the curve object and make it the active object
            tree.select_set(True)
            bpy.context.view_layer.objects.active = tree

            # Go into edit mode
            bpy.ops.object.mode_set(mode='EDIT')

            # Deselect all points in the curve
            bpy.ops.curve.select_all(action='DESELECT')
            for stem in range(seed['tree_parameters']['baseSplits']+1):
                for point in tree.data.splines[stem].bezier_points:
                    point.select_control_point = True

            # Separate the selected points into a new object
            bpy.ops.curve.separate()

            # Go back to object mode
            bpy.ops.object.mode_set(mode='OBJECT')

            bpy.data.objects['temporary'].name      = 'branches'
            bpy.data.objects['temporary.001'].name  = 'stem'  
        
    def spawn_undergrowth_sapling_sets(self,n_sets=15,set_size=25,size=4,dist=5,remove_after_save=False,showLeaves=True):
        def plant_sapling(size=2, sizeV=0.5, stem=1, stemV=0.1, bevel=True, showLeaves=True, seed=0, curveRes=(10, 8, 3, 1), curve=(0, 30, 25, 0), curveV=(10, 10, 25, 0), curveBack=(0, 0, 0, 0), rootFlare=1.15, ratio=0.02, minRadius=0.002, baseSplits=2, segSplits=(0.35, 0.35, 0.35, 0), splitByLen=True, rMode='rotate', splitStraight=0, splitLength=0, splitAngle=(20, 36, 32, 0), splitAngleV=(2, 2, 0, 0), splitHeight=0.5, splitBias=0, baseSize=0.35, baseSize_s=0.8, branchDist=1.5, nrings=-1, levels=2, branches=(0, 20, 10, 5), length=(1, 0.33, 0.375, 0.45), lengthV=(0.05, 0.2, 0.35, 0), attractUp=(0, -1, -0.65, 0), attractOut=(0, 0.2, 0.25, 0), shape='8', shapeS='7', customShape=(1, 1, 0.5, 1), downAngle=(90, 60, 50, 45), downAngleV=(0, 25, 30, 10), useOldDownAngle=False, useParentAngle=True, rotate=(99.5, 137.5, 137.5, 137.5), rotateV=(0, 0, 0, 0), leaves=16, leafType='0', leafDownAngle=45, leafDownAngleV=10, leafRotate=137.5, leafRotateV=0, leafObjZ='+2', leafObjY='+1', leafScale=0.2, leafScaleX=0.5, leafScaleT=0.2, leafScaleV=0.25, leafShape='hex', leafangle=-45, leafDist='6', leafBaseSize=0.2, closeTip=True, noTip=False, autoTaper=True, taper=(1, 1, 1, 1), radiusTweak=(1, 1, 1, 1), ratioPower=1, attachment='0'):
            tree = bpy.ops.curve.tree_add(
                                        # General Props
                                        bevel=bevel, 
                                        showLeaves=showLeaves, 
                                        seed=seed, 
                                        curveRes=curveRes, 
                                        curve=curve, 
                                        curveV=curveV, 
                                        curveBack=curveBack,
                                        
                                        # Trunk Props
                                        scale=size, 
                                        scaleV=sizeV, 
                                        scale0=stem, 
                                        scaleV0=stemV, 
                                        rootFlare=rootFlare, 
                                        ratio=ratio, 
                                        minRadius=minRadius,    
                                        baseSplits=baseSplits, 
                                        segSplits=segSplits, 
                                        splitByLen=splitByLen, 
                                        rMode=rMode, 
                                        splitStraight=splitStraight, 
                                        splitLength=splitLength,
                                        splitAngle=splitAngle, 
                                        splitAngleV=splitAngleV, 
                                        splitHeight=splitHeight, 
                                        splitBias=splitBias,
                                        baseSize=baseSize, 
                                        baseSize_s=baseSize_s, 
                                        branchDist=branchDist, 
                                        nrings=nrings, 

                                        # Branch Props
                                        levels=levels, 
                                        branches=branches, 
                                        length=length, 
                                        lengthV=lengthV,
                                        attractUp=attractUp, 
                                        attractOut=attractOut, 
                                        shape=shape, 
                                        shapeS=shapeS, 
                                        customShape=customShape, 
                                        downAngle=downAngle, 
                                        downAngleV=downAngleV, 
                                        useOldDownAngle=useOldDownAngle, 
                                        useParentAngle=useParentAngle, 
                                        rotate=rotate,
                                        rotateV=rotateV,

                                        # Leaf Props
                                        leaves=leaves, 
                                        leafType=leafType, 
                                        leafDownAngle=leafDownAngle, 
                                        leafDownAngleV=leafDownAngleV, 
                                        leafRotate=leafRotate, 
                                        leafRotateV=leafRotateV, 
                                        leafObjZ=leafObjZ, 
                                        leafObjY=leafObjY, 
                                        leafScale=leafScale, 
                                        leafScaleX=leafScaleX, 
                                        leafScaleT=leafScaleT, 
                                        leafScaleV=leafScaleV, 
                                        leafShape=leafShape, 
                                        leafangle=leafangle, 
                                        leafDist=leafDist,
                                        leafBaseSize=leafBaseSize, 
                                        closeTip=closeTip, 
                                        noTip=noTip,
                                        
                                        # Rest
                                        autoTaper=autoTaper, 
                                        taper=taper, 
                                        radiusTweak=radiusTweak, 
                                        ratioPower=ratioPower,
                                        attachment=attachment
                                    )

        def r(n, r0=0.2, rinf=0.1, k=0.5):
            if rinf < r0:
                return rinf+((r0-rinf)*np.exp(-k*n))
            else:
                return rinf-((rinf-r0)*np.exp(-k*n))      

        
        self.parameters['undergrowth_sapling_sets']      = n_sets
        self.parameters['undergrowth_sapling_set_size']  = set_size
        
        obs     = []
        rnd_arr = np.random.rand(n_sets,2)
        
        for x,y in rnd_arr:
            x = (x*self.parameters['extend_x']) - (self.parameters['extend_x']/2)
            y = (y*self.parameters['extend_y']) - (self.parameters['extend_y']/2)

            for _ in range(set_size):
                [shift_x,shift_y] = np.random.normal(0, dist, 2)
                location            = self.calculate_z([x+shift_x,y+shift_y])
                
                if -self.parameters['extend_x']/2 <= location[0] <= self.parameters['extend_x']/2 and -self.parameters['extend_y']/2 <= location[1] <= self.parameters['extend_y']/2:

                    if not any([self.vertices[n_idx]['occupied'] for (_,n_idx,_) in self.kdTrees['kd_3D'].find_range(location,radius=0.2)]):
                        d = (shift_x**2+shift_y**2)**(1/2)
                        
                        s = r(d,r0 = size,rinf=0.3,k=0.2) 

                        shape       = (1, 1, 0.5, 1)
                        leaves      = 16 
                        branches    = (0,20,10,0)
                        baseSplits  = np.random.randint(1,5)
                        tree    = plant_sapling(size=s,
                                                seed=np.random.randint(1000),
                                                showLeaves=showLeaves,
                                                baseSplits=baseSplits,
                                                customShape=(r(d,shape[0]/3,shape[0],k=0.75),
                                                             r(d,shape[1]/3,shape[1],k=0.75),
                                                             r(d,shape[2],shape[2]/2,k=0.75),
                                                             r(d,shape[3]/3,shape[3],k=0.75)),
                                                leaves = int(np.ceil(r(d,leaves,0.1,k=2))),
                                                branches = (branches[0],
                                                            int(np.ceil(r(d,branches[1],branches[1]/2))),
                                                            int(np.ceil(r(d,branches[2],branches[2]/2))),
                                                            branches[3]) 
                                                )
                        
                        ## Leaves
                        if showLeaves:
                            # Deselect all points in the curve
                            bpy.ops.object.select_all(action='DESELECT')

                            # Select in data
                            leaves = bpy.data.objects['leaves']

                            leaves.name = 'broadleafs'

                            # Select the curve object and make it the active object
                            leaves.select_set(True)
                            bpy.context.view_layer.objects.active = leaves

                            # Clear Parent
                            bpy.ops.object.parent_clear(type='CLEAR')
                            
                            leaves.location   = location
                            
                        ## Stem & Branches & Twigs
                        # Deselect all points in the curve
                        bpy.ops.object.select_all(action='DESELECT')

                        # Select in data
                        tree = bpy.data.objects['tree']

                        tree.name       = 'temporary'
                        tree.location   = location

                        # Select the curve object and make it the active object
                        tree.select_set(True)
                        bpy.context.view_layer.objects.active = tree

                        # Go into edit mode
                        bpy.ops.object.mode_set(mode='EDIT')

                        # Deselect all points in the curve
                        bpy.ops.curve.select_all(action='DESELECT')
                        for stem in range(baseSplits+1):
                            for point in tree.data.splines[stem].bezier_points:
                                point.select_control_point = True

                        # Separate the selected points into a new object
                        bpy.ops.curve.separate()

                        # Go back to object mode
                        bpy.ops.object.mode_set(mode='OBJECT')

                        bpy.data.objects['temporary'].name      = 'branches'
                        bpy.data.objects['temporary.001'].name  = 'stem'

    def combine_objects(self,remove_after_save=False):
        for obj_name in self.objects['names']:
            bpy.ops.object.select_all(action='DESELECT')
            for obj in bpy.data.objects:
                if obj_name in obj.name:
                    ob = obj
                    bpy.context.view_layer.objects.active = ob
                    ob.select_set(True)
            bpy.ops.object.join()
            bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')            
            ob.name = obj_name
        
            file_path = f'{self.path["scene_output"]["objects"]}\\{ob.name}.obj'
            self.helios['object_files'].append(file_path)  
            if self.parameters['save_outputs']:                      
                bpy.ops.export_scene.obj(
                            filepath=file_path,
                            use_selection=True,
                            axis_up='Z',
                        )
                if remove_after_save:
                    bpy.data.objects.remove(ob, do_unlink=True)
                
    # -----------------------------------------------------------------
    # Scanning: -------------------------------------------------------
    # ----------------------------------------------------------------- 
                   
    def create_path_DEM(self,subs=10):
        # Get the active object
        dem = self.objects['DEM']
        
        bpy.context.view_layer.objects.active = dem
        
        obj = dem.copy()
        obj.data = dem.data.copy()
        
        bpy.context.collection.objects.link(obj)
        
        obj.name = 'path_DEM'

        # Make sure you're in object mode
        bpy.ops.object.mode_set(mode='OBJECT')

        # Create a BMesh from the active object's mesh data
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()

        # Get the vertices you want to delete. This example deletes the first vertex.
        verts_to_delete = []
        for i,key in enumerate(self.vertices.keys()):
            if not self.vertices[key]['accessible']:
                verts_to_delete.append(bm.verts[i])

        # Delete the vertices
        bmesh.ops.delete(bm, geom=verts_to_delete, context='VERTS')

        # Write the BMesh back to the mesh data
        bm.to_mesh(obj.data)
        bm.free()
        
        # Set the DEM as the active object
        bpy.context.view_layer.objects.active = obj

        # Make sure the DEM is selected
        obj.select_set(True)

        # Set modifier
        obj.modifiers.new("deci", 'DECIMATE')

        # Set the modifier properties
        mod                 = obj.modifiers["deci"]
        mod.decimate_type   = 'UNSUBDIV'
        mod.iterations      = 10
        
        # Apply modifier
        bpy.ops.object.modifier_apply(modifier="deci",single_user=True)
        
        
        self.objects['path_DEM'] = obj        
        # Update the mesh
        obj.data.update()
        
    def create_path_graph(self,n_neighbours=10):
        # select path_DEM object
        obj = self.objects['path_DEM']
        
        # Change to Object-Mode
        bpy.ops.object.mode_set(mode='OBJECT')
                
        self.kdTrees['kd_path'] = mathutils.kdtree.KDTree(len(obj.data.vertices))
        
        for idx,v in enumerate(obj.data.vertices): 
            self.kdTrees['kd_path'].insert([v.co.x,v.co.y,v.co.z], idx)
            
        self.kdTrees['kd_path'].balance()
        
        self.path_graph = dict()
        for v in obj.data.vertices:
            self.path_graph[v.index] = dict()
            for _,idx,d in self.kdTrees['kd_path'].find_n(v.co,n_neighbours):
                if idx > 0:
                    self.path_graph[v.index][idx] = d 
    
    def get_path(self,start_loc=(0,0),end_loc=(25,-25),method='dijkstra',append_path=True):
        
        start_loc   = self.calculate_z(start_loc)
        end_loc     = self.calculate_z(end_loc)
        
        if method == 'dijkstra':
            path = self.dijkstra_pathfinding(start_loc,end_loc)
        
        obj = self.objects['path_DEM']
        if append_path:
            for idx in path:
                 self.scan_path.append([obj.data.vertices[idx].co.x,obj.data.vertices[idx].co.y,obj.data.vertices[idx].co.z+1])
        else:
            self.scan_path = []
            for idx in path:
                 self.scan_path.append([obj.data.vertices[idx].co.x,obj.data.vertices[idx].co.y,obj.data.vertices[idx].co.z+1])  
            
    def dijkstra_pathfinding(self,start_loc,end_loc):
        _,start,_ = self.kdTrees['kd_path'].find(start_loc)
        _,end,_ = self.kdTrees['kd_path'].find(end_loc) 
        
        shortest_paths = {start: (None, 0)}
        current_node = start
        visited = set()

        while current_node != end:
            visited.add(current_node)
            destinations = self.path_graph[current_node]
            weight_to_current_node = shortest_paths[current_node][1]

            for next_node, weight in destinations.items():
                weight = weight + weight_to_current_node
                if next_node not in shortest_paths:
                    shortest_paths[next_node] = (current_node, weight)
                else:
                    current_shortest_weight = shortest_paths[next_node][1]
                    if current_shortest_weight > weight:
                        shortest_paths[next_node] = (current_node, weight)

            next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
            if not next_destinations:
                return "Route Not Possible"
            current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

        path = []
        while current_node is not None:
            path.append(current_node)
            next_node = shortest_paths[current_node][0]
            current_node = next_node
        path = path[::-1]
        return path
    
    def simulate_path(self,velocity=2,framerate=30,save_gps=True):
        v = velocity/3.6/framerate # meters per frame
        
        # Create an icosphere
        bpy.ops.mesh.primitive_ico_sphere_add(location=(0, 0, 0))
        ico_sphere = bpy.context.object


        # Define the time step between each coordinate (in frames)
        frame           = 0
        self.gps_track  = []
        
        # Loop through the coordinates
        for i, coord in enumerate(self.scan_path):
            if i == 0:
                # Set the icosphere's location to the current coordinate
                ico_sphere.location = coord
                # Insert a keyframe at the current frame
                ico_sphere.keyframe_insert(data_path="location", frame=0)
            else:
                d       = ((coord[0]-self.scan_path[i-1][0])**2+
                           (coord[1]-self.scan_path[i-1][1])**2+
                           (coord[2]-self.scan_path[i-1][2])**2)**(1/2)
                frame   += d/v 
                ico_sphere.location = coord
                ico_sphere.keyframe_insert(data_path="location", frame=frame)
                
            self.gps_track.append(coord)
            
    def create_leg_4Helios_MLS(self,rpmRoll=26,vWalk=1.5,turnTime=0.001):
        
        self.helios['legs'] = ['#TIME_COLUMN: 0','#HEADER: "t", "pitch", "roll", "yaw", "x", "y", "z"']
        t       = 0
        roll    = 0
        yaw     = 0

        for i in range(len(self.gps_track)):
            if i == 0:
                # starting point on Tripod (pitch = 0)
                pos     = self.gps_track[i]
                t       = 0
                roll    = 0
                pitch   = 105
                yaw     = 0
                x       = pos[0]
                y       = pos[1]
                z       = pos[2]
                self.helios['legs'].append(f'{t}, {pitch}, {roll}, {yaw}, {-y}, {x}, {z}')

                # turning towards next pos:
                posNext = self.gps_track[i + 1]
                dx      = posNext[0]-pos[0]
                dy      = posNext[1]-pos[1]
                if dy == 0 and dx < 0:
                    yaw = 270
                elif dy == 0 and dx >= 0:
                    yaw = 90
                else:
                    yaw = np.degrees(np.arctan(dx/dy))
                    if yaw < 0:
                        yaw = 360 + yaw
                t       += turnTime
                self.helios['legs'].append(f'{t}, {pitch}, {roll}, {yaw}, {-y}, {x}, {z}')

            elif i == len(self.gps_track)-1:
                # arriving at end pos:
                pos     = self.gps_track[i]
                posOld  = self.gps_track[i-1]
                dx      = posOld[0]-pos[0]
                dy      = posOld[1]-pos[1]
                l       = (dx**2 + dy**2)**(1/2)
                dt      = l/vWalk
                dRoll   = (360*(rpmRoll/60))*dt
                t       += dt
                roll    += dRoll
                pitch   = 90
                yaw     += 0
                x       = pos[0]
                y       = pos[1]
                z       = pos[2]
                self.helios['legs'].append(f'{t}, {pitch}, {roll}, {yaw}, {-y}, {x}, {z}')

            else:
                # arriving at intermediate pos:
                pos     = self.gps_track[i]
                posOld  = self.gps_track[i-1]
                dx      = posOld[0]-pos[0]
                dy      = posOld[1]-pos[1]
                l       = (dx**2 + dy**2)**(1/2)
                dt      = l/vWalk
                dRoll   = (360*(rpmRoll/60))*dt
                t       += dt
                roll    += dRoll
                pitch   = 90
                yaw     += 0
                x       = pos[0]
                y       = pos[1]
                z       = pos[2]
                self.helios['legs'].append(f'{t}, {pitch}, {roll}, {yaw}, {-y}, {x}, {z}')

                # turning towards next pos:
                posNext = self.gps_track[i+1]
                dx      = posNext[0]-pos[0]
                dy      = posNext[1]-pos[1]
                if dy == 0 and dx < 0:
                    yaw = 270
                elif dy == 0 and dx >= 0:
                    yaw = 90
                else:
                    yaw = np.degrees(np.arctan(dx/dy))
                    if yaw < 0:
                        yaw = 360 + yaw
                t       += turnTime
                self.helios['legs'].append(f'{t}, {pitch}, {roll}, {yaw}, {-y}, {x}, {z}')
                
        self.path['scene_output']['trajectory'] = f'{self.path["scene_output"]["helios"]}\\MLS.trj'  
              
        with open(self.path['scene_output']['trajectory'], mode="wt") as f:
            for line in self.helios['legs']:
                f.write(f'{line}\n')
                
    def write_xml_4Helios(self,scanner_path='data\\scanners_tls.xml',
                          scanner_name='vlp16',head_rot_speed = 3600,pulse_freq=180000):
                
        self.helios['scanner_path']     = scanner_path
        self.helios['scanner']          = scanner_name
        self.helios['scene_name']       = f'scene_nr_{str(self.parameters["scene_nr"]).zfill(4)}'
        self.helios['scene_path']       = f'{self.path["scene_output"]["helios"]}\\scene.xml'
        self.helios['survey_path']      = f'{self.path["scene_output"]["helios"]}\\survey.xml'


        surveyXML  = [f'<?xml version="1.0" encoding="UTF-8"?>',
                      f'<document>',
                      f'\t<scannerSettings id="scaset" active="true" pulse_freq_hz="{pulse_freq}" scanFreq_hz="100"/>',
                      f'\t<survey name="{self.helios["scene_name"]}" ',
                      f'\t\tscene="{self.helios["scene_path"]}#{self.helios["scene_name"]}" ',
                      f'\t\tplatform="interpolated" basePlatform="data/platforms.xml#sr22" ',
                      f'\t\tscanner="{self.helios["scanner_path"]}#{self.helios["scanner"]}">'
                      ]

        for l in range(3,len(self.helios['legs'])):
            if l > 0:
                if l == 3:
                    head_rot_start = float(self.helios['legs'][l-1].split(",")[0])*head_rot_speed
                    head_rot_stop  = float(self.helios['legs'][l].split(",")[0])*head_rot_speed
                    surveyXML += [f'\t\t<leg>',
                                  f'\t\t\t<platformSettings ',
                                  f'\t\t\t\ttrajectory="{self.path["scene_output"]["trajectory"]}"',
                                  f'\t\t\t\ttIndex="0" xIndex="4" yIndex="5" zIndex="6" ',
                                  f'\t\t\t\trollIndex="2" pitchIndex="1" yawIndex="3"',
                                  f'\t\t\t\tslopeFilterThreshold="0.0" toRadians="true" syncGPSTime="true"',
                                  f'\t\t\t\ttStart="{self.helios["legs"][l-1].split(",")[0]}"',
                                  f'\t\t\t\ttEnd="{self.helios["legs"][l].split(",")[0]}"',
                                  f'\t\t\t\tteleportToStart="true"',
                                  f'\t\t\t/>',
                                  f'\t\t\t<scannerSettings template="scaset"',
                                  f'\t\t\t\theadRotatePerSec_deg="{head_rot_speed}"',
                                  f'\t\t\t\theadRotateStart_deg="{head_rot_start}" ',
                                  f'\t\t\t\theadRotateStop_deg="{head_rot_stop}"',
                                  f'\t\t\t\ttrajectoryTimeInterval_s="0.01"/>',
                                  f'\t\t</leg>'
                                  ]
                else:
                    head_rot_start = float(self.helios['legs'][l-1].split(",")[0])*head_rot_speed
                    head_rot_stop  = float(self.helios['legs'][l].split(",")[0])*head_rot_speed
                    surveyXML += [f'\t\t<leg>',
                                  f'\t\t\t<platformSettings ',
                                  f'\t\t\t\ttrajectory="{self.path["scene_output"]["trajectory"]}"',
                                  f'\t\t\t\ttStart="{self.helios["legs"][l-1].split(",")[0]}"',
                                  f'\t\t\t\ttEnd="{self.helios["legs"][l].split(",")[0]}"',
                                  f'\t\t\t\tteleportToStart="true"',
                                  f'\t\t\t/>',
                                  f'\t\t\t<scannerSettings template="scaset"',
                                  f'\t\t\t\theadRotatePerSec_deg="{head_rot_speed}"',
                                  f'\t\t\t\theadRotateStart_deg="{head_rot_start}" ',
                                  f'\t\t\t\theadRotateStop_deg="{head_rot_stop}"',
                                  f'\t\t\t\ttrajectoryTimeInterval_s="0.01"/>',
                                  f'\t\t</leg>'
                                  ]

        surveyXML += [f'\t</survey>',
                      f'</document>']

        sceneXML = [f'<?xml version="1.0" encoding="UTF-8"?>',f'<document>',
                    f'\t<scene id="{self.helios["scene_name"]}" name="{self.helios["scene_name"]}">']
        
        self.object_labels = {}
        for label,file in enumerate(self.helios['object_files']):
            self.object_labels[int(label)] = file.split("\\")[-1][:-4]
            sceneXML += [f'\t\t<part>',
                         f'\t\t\t<filter type="objloader">',
                         f'\t\t\t\t<param type="string" key="filepath" value="{file}" />',
                         f'\t\t\t</filter>',
                         f'\t\t</part>']

        sceneXML += [f'\t</scene>',f'</document>']                                               

        with open(self.helios['survey_path'], mode="wt") as t:
            for line in surveyXML:
                t.write(f'{line}\n')
                
        with open(self.helios['scene_path'], mode="wt") as t:
            for line in sceneXML:
                t.write(f'{line}\n')

    def run_Helios(self,flags=''):
        self.helios['path'] = self.path['helios']
        subprocess.run([f'{self.helios["path"]}run/helios.exe',
                        f'{str(self.helios["survey_path"])}',
                        f'{flags}'],
                       cwd=self.helios['path'])

        #self.combine_Helios_legs()
    def label_converter(self,l,combined_classes=[{'label':1,'keywords':['ground','rock']},
                                                 {'label':2,'keywords':['plant','leaves','branches','needles','leaf','sapling']},
                                                 {'label':3,'keywords':['dw','stump']},
                                                 {'label':4,'keywords':['birch','pine','maple','stem']}]):
        cla = -1                                         
        for classes in combined_classes:
            if any([keyword in self.object_labels[l] for keyword in classes['keywords']]):
                cla = classes['label']
        return cla
        



    def combine_Helios_legs(self,removeLegs=True,save_combined=True,create_chunks=True,chunk_size=[5,5,2],center=(0,0),extend=(25,25),max_height=2):
        def deleteSubfolderAndFiles(folderPath):
            for entry in os.listdir(folderPath):
                if os.path.isdir(os.path.join(folderPath,entry)):
                    deleteSubfolderAndFiles(os.path.join(folderPath,entry))
                    os.rmdir(os.path.join(folderPath,entry))
                else:
                    os.remove(os.path.join(folderPath,entry))
        
        if create_chunks:
            chunk_dict = dict()
            
        if save_combined:
            combined = []
        
        self.helios['helios_output_dir'] = self.path['scene_output']['pointclouds']
        
        folders = [os.path.join(f'{self.helios["path"]}output', entry) for entry in os.listdir(f'{self.helios["path"]}output')
                   if os.path.isdir(os.path.join(f'{self.helios["path"]}output', entry))]
                   
        for folder in folders:
            if str(self.parameters["scene_nr"]).zfill(4) in folder:
                subFolders  = [os.path.join(f'{folder}', entry) for entry in os.listdir(f'{folder}')
                               if os.path.isdir(os.path.join(f'{folder}', entry))]
                for subFolder in subFolders:
                    xyzFiles    = [os.path.join(f'{subFolder}', file) for file in os.listdir(subFolder) if file.endswith('.xyz')]
                    for xyzFile in xyzFiles:
                        with open(xyzFile,'r') as f:
                            for line in f:   
                                l = line.split(' ')
                                
                                x_glob      = float(l[0])                         
                                y_glob      = float(l[1])      
                                z_glob      = float(l[2])
                                label       = int(l[8])
                
                                [_,_,z_dem] = self.calculate_z([y_glob,-x_glob])
                                z_norm      = np.round(z_glob-z_dem,6)
                                
                                label       = int(l[8])                            
                                point2tree  = self.label_converter(label)  
                                   
                                
                                             
                                if create_chunks:  
                                    x_chunk     = int(np.floor(x_glob/chunk_size[1]))
                                    y_chunk     = int(np.floor(y_glob/chunk_size[0]))
                                    
                                    x_loc       = np.round(x_glob%chunk_size[0],6)
                                    y_loc       = np.round(y_glob%chunk_size[1],6)
                                    
                                    z_norm      = np.round(z_glob-z_dem,6)                                                                    
                                    if z_norm <= chunk_size[2]:
                                        chunk_name = f'chunk_{x_chunk}_{y_chunk}'
                                        
                                        if chunk_name not in chunk_dict.keys():
                                            chunk_dict[chunk_name] = []
                                        
                                        chunk_dict[chunk_name].append(f'{x_loc} {y_loc} {z_norm} {point2tree}\n')   
                                        
                                if save_combined:
                                    combined.append(f'{x_glob} {y_glob} {z_glob} {z_norm} {point2tree} {label}')
                 
   
            if removeLegs:
                deleteSubfolderAndFiles(folder)    

        if save_combined:
            with open(f'{self.helios["helios_output_dir"]}/combined.xyz','a') as c:
                c.writelines(combined)
        
        if create_chunks:
            os.mkdir(f'{self.helios["helios_output_dir"]}/chunks/')
            for chunk in chunk_dict.keys():
                
                np.save(f'{self.helios["helios_output_dir"]}/chunks/{chunk}.npy',np.array([line.rstrip().split(' ') for line in chunk_dict[chunk]]).astype(float)) 
                
                '''
                with open(f'{self.helios["helios_output_dir"]}/chunks/{chunk}.xyz','a') as c:
                    c.writelines(chunk_dict[chunk])
                '''       
                                            
    # -----------------------------------------------------------------
    # Pipelines: ------------------------------------------------------
    # -----------------------------------------------------------------                         

    def map_based_pipeline(self,DEM=None,CHM=None,leaf_type_map=None,n_trees=70,n_rocks=5,n_stumps=5,
                           n_sets_undergrowth=20,set_size_undergrowth=50,size_undergrowth=3,dist_undergrowth=5,
                           n_laying_dw=10,n_plants=5,showLeaves=True,type_threshold=0.5,
                           n_x=200,n_y=200,d_x=0.1,d_y=0.1,d_z=1,max_height=35,boundry=0,
                           create_rnd=True,n_layer=5,std_dev=5.0,tree_decimate=np.pi/18):
        # Adding plot information
        print(f'Start Scene Nr {self.parameters["scene_nr"]}')
        print('')
        print('Add DEM')
        
        self.add_DEM(DEM=DEM,create_rnd=create_rnd,n_layer=n_layer,std_dev=std_dev,n_x=n_x,n_y=n_y,d_x=d_x,d_y=d_y,d_z=d_z,boundry=boundry)
        print('finished')
        print('')
        print('Add CHM')
        self.add_CHM(CHM=CHM,create_rnd=create_rnd,n_layer=n_layer,std_dev=std_dev,max_height=max_height)
        print('finished')
        print('')
        print('Add Dominante Leaf Type')
        self.add_dominante_leaf_type(leaf_type_map=leaf_type_map,type_threshold=type_threshold,create_rnd=create_rnd,n_layer=n_layer,std_dev=std_dev)
        print('finished')
        print('')
        print('Add Ground Vegetation')
        self.add_ground_vegetation_map()
        print('finished')
        print('')
        
        # Adding ground objects that occupy space
        print('Spawn laying Deadwood')
        self.spawnDeadWoodStemSamples(self.path['laying_dw'],n_stems=n_laying_dw)
        print('finished')
        print('')
        print('Spawn Rocks')        
        self.spawnRocks(self.path['rocks'],n_rocks=n_rocks)
        print('finished')
        print('')
        print('Spawn Stumps')
        self.spawnStumps(self.path['tree_stumps'],n_stumps=n_stumps,radius=1,lower_factor=1,elevate_factor=1/4)
        print('finished')
        print('')
        print('Spawn Plants')
        self.spawnBroadLeafedPlant(objpath=f'{self.path["ground_veg"]}\\broadLeafPlant.obj',n_sets=n_plants,set_size=15) 
        print('finished')
        print('')   
        
        # Plant trees
        print('Seed Trees')
        self.seed_rnd_trees(n_trees=n_trees)
        print('finished/n')
        print('Calculate Neighbourhood')
        self.get_neighbour_information()
        print('finished/n')
        print('Grow Trees')
        self.grow_trees(showLeaves=showLeaves,decimate=tree_decimate)
        print('finished/n')
        
        # Plant understory trees
        print('Seed Understory Trees')
        self.seed_understory_trees()
        print('finished/n')
        print('Grow Understory Trees')
        self.grow_understory_trees()
        print('finished/n')
        print('Spawn Undergrowth')
        self.spawn_undergrowth_sapling_sets(n_sets=n_sets_undergrowth,set_size=set_size_undergrowth,
                                            size=size_undergrowth,dist=dist_undergrowth)
        print('finished/n')
        
        # Combine individual objects
        self.combine_objects()
         
        # Export vertices
        print('Export Vertices')
        self.export_vertices()
        print('finished/n')
        
    def simulate_walk(self,coords=[[0,0],[-1,-1],[1,-1],[1,-0.5],[-1,-0.5],[-1,0],[1,0],[1,0.5],[-1,0.5],[-1,1],[1,1],[0,0]],distance_factor=25):
        print('Create Path DEM')
        self.create_path_DEM()
        print('finished/n')
        print('Create Path Graph')
        self.create_path_graph()
        print('finished/n')
        print('Get Shortest Paths')
        
        for i in range(len(coords)-1):
            self.get_path(start_loc=np.array(coords[i])*distance_factor,
                          end_loc=np.array(coords[i+1])*distance_factor)
            
        print('finished/n')
        print('Simulate Path')
        self.simulate_path()
        print('finished/n')
        
    def simulate_scan(self,removeLegs=True,save_combined=True,
                      create_chunks=True,chunk_size=[5,5,2],center=(0,0),extend=(30,30),max_height=2):
        if not self.parameters['save_outputs']:
            print('Simulating the scan only works after saving the outputs.\nAdd "save_objects=True" when initializing the class.')
        else:
            print('Create Helios Legs')
            self.create_leg_4Helios_MLS()
            print('finished/n')
            print('Prepare XML Files')
            self.write_xml_4Helios()
            print('finished/n')
            print('Run Helios')
            self.run_Helios()
            print('finished/n')
            print('Combine Legs')
            self.combine_Helios_legs(removeLegs=removeLegs,center=center,extend=extend,max_height=max_height,
                                     create_chunks=create_chunks,chunk_size=chunk_size,save_combined=save_combined)
            print('finished/n')
        
    def clean_scene(self):
        print('Clean all')
        # Unlink all objects
        for obj in bpy.context.scene.objects:
            if obj.name != 'Sun':
                bpy.context.scene.collection.objects.unlink(obj)

        # Remove unused data blocks
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)

        for block in bpy.data.materials:
            if block.users == 0:
                bpy.data.materials.remove(block)

        for block in bpy.data.textures:
            if block.users == 0:
                bpy.data.textures.remove(block)

        for block in bpy.data.images:
            if block.users == 0:
                bpy.data.images.remove(block)               

    # -----------------------------------------------------------------
    # Work in Progress: -----------------------------------------------
    # -----------------------------------------------------------------  
    def add_tree_by_dict(self,tree_dict):
        if type(tree_dict) == dict:
            for tree in tree_dict:
                location = calculate_z((tree['x'],tree['y']))
                
                pos,id,_ = self.kdTrees['kd_3D'].find(location)
                
                self.vertices[id]['tree'] = tree 

 
        
# ---------------------------------------------------------------------
#                         Running Scripts
# ---------------------------------------------------------------------
scene = worldGenerator(path)

scene.map_based_pipeline(DEM=None,
                         CHM=None,
                         leaf_type_map=None,
                         n_trees=90,
                         n_rocks=5,
                         n_stumps=10,
                         n_laying_dw=60,
                         n_plants=50,
                         n_sets_undergrowth=5,
                         set_size_undergrowth=15,
                         size_undergrowth=3,
                         dist_undergrowth=5,
                         showLeaves=True,
                         type_threshold=0.5,
                         n_x=550,
                         n_y=550,
                         d_x=0.1,
                         d_y=0.1,
                         d_z=5,
                         max_height=35,
                         boundry=2,
                         create_rnd=True,
                         n_layer=5,
                         std_dev=5.0)
                         
scene.simulate_walk(distance_factor=25)
scene.simulate_scan()

