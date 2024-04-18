import bpy
import numpy as np

# Importing Objects
def import_obj(filepath, location=(0.0,0.0,0.0),rotation=(0.0,0.0,0.0),scale=(1.0,1.0,1.0)):
    # Import the .obj file
    bpy.ops.import_scene.obj(filepath=filepath)
    
    # Get the imported object
    obj = bpy.context.selected_objects[0]
    
    # Set the object's location
    obj.location        = location
    obj.rotation_euler  = rotation
    obj.scale           = scale
    
    obj.select_set(True)
    
    return obj


    
def bend_obj(obj,degree):
    bpy.ops.object.empty_add(type='SPHERE', align='WORLD', location=(0, 0, 0), scale=(0.1, 0.1, 0.1))
    empty = bpy.context.object
    empty.rotation_euler = (np.radians(90), np.radians(90), 0)

    # Ensure the object has a SimpleDeform modifier
    if "SimpleDeform" not in obj.modifiers:
        obj.modifiers.new("SimpleDeform", 'SIMPLE_DEFORM')

    # Set the modifier properties
    modifier = obj.modifiers["SimpleDeform"]
    modifier.deform_method = 'BEND'
    modifier.origin = empty
    modifier.deform_axis = 'X'
    modifier.angle = np.radians(degree)
    
    return obj,empty
    
def generateRnd(filepath,output_folder,normal_loc=0,normal_scale=3,max_bend=20,delete_obj=True):
    filename = filepath.split('\\')[-1]
    parts = filename.split('_')
    for part in parts:
        if part.startswith('d'):
            d = int(part[1:4])
        elif part.startswith('l'):
            l = int(part[1:5])
        else:
            name = part
            
    s       = np.abs(np.random.normal(normal_loc,normal_scale,1))
    if s < 0.1:
        s = 0.1

    l_new   = int(s*l)
    d_new   = int(s*d)

    bend_angle  = np.random.random(1)*max_bend*2-max_bend
    rot_angle   = np.random.random(1)*360
     
    obj         = import_obj(filepath=filepath,location = (0,0,0),rotation = (0,np.radians(rot_angle),0),scale=(s,s,s))
    obj.name    = f'{name}_l{str(l_new).zfill(4)}_d{str(d_new).zfill(3)}'
    obj,empty   = bend_obj(obj,bend_angle)

    file_new = f'{output_folder}\\{name}_l{str(l_new).zfill(4)}_d{str(d_new).zfill(3)}.obj'
    
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.ops.export_scene.obj(
                filepath=file_new,
                use_selection=True,
                axis_up='Z',
            )
    if delete_obj:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.objects.remove(empty, do_unlink=True)   


output_folder = 'H:\\Blender\\WorldGeneration\\Assets\\Deadwood'
filepath = "H:\\Blender\\WorldGeneration\\Assets\\NormalizedBuildingBlocks\\Deadwood\\stem_l0100_d010.obj"
for _ in range(30):
    generateRnd(filepath ,output_folder,delete_obj=False)
    