import bpy
import numpy as np

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
            

def check_crown_diameter(name='tree'):
    max_d = 0

    for spline in bpy.data.objects[name].data.splines:
        for point in spline.bezier_points:
            d = (point.co[0]**2+point.co[1]**2)**(1/2)
            if d > max_d:
                max_d = d
                
    return max_d
