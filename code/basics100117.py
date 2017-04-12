##### ./blender -E CYCLES --python scripts/basics.py


# bpy is high level api for blender python interace
import bpy
import numpy as np
import random
import math
import sys
from mathutils import *

scn = bpy.context.scene
scn.render.engine = 'CYCLES'
scn.world.use_nodes = True

#individual parameters
# data_location='/home/kiwi/blender/scripts'
# outputpath = 'scripts/output'
data_location = 'c:/Blender/draft_main'
outputpath = 'scripts/output/init'

# lets remove the cube and design our scene
for object in bpy.data.objects:
    if object.type == 'MESH':
        object.select = True
    else:
        object.select = False
bpy.ops.object.delete()

## removing objects from the scene doesnot necessarily clear the cache
for mesh in bpy.data.meshes:
    bpy.data.meshes.remove(mesh)
for material in bpy.data.materials:
    bpy.data.materials.remove(material)

######----------------Camera settings---------------------
ground_radius=20

def CameraSettings(loc,rot,lensU,lens):
    ## one can add the cameras by using
    # bpy.ops.object.camera_add()

    ## set extrinsic parameters of the camera
    # TODO: motion blur, illumination, specular reflection
    bpy.data.objects['Camera'].location = loc
    bpy.data.objects['Camera'].rotation_euler = rot
    
    ## camera intrisic parameters
    bpy.data.cameras['Camera'].lens_unit = lensU #'MILLIMETERS' # 'FOV'
    bpy.data.cameras['Camera'].lens = lens

## one can also code PSF, or camera trasfer function and set as follows
#bpy.ops.scripts.python_file_run(filepath='your script location')

#CameraSettings((-30,2,5),(90*math.pi/180.0,0.0,-90.0*math.pi/180.),'FOV',40.0) #'MILLIMETERS' # 'FOV'
if random.uniform(0.0,1.0) > 0.5:
    camera_loc = ((random.uniform(0.0,15.0)),random.uniform(0.0,1.0),random.uniform(3,7))
    camera_rot = (random.uniform(60.0,75.0)*math.pi/180.,0.,(random.uniform(0.0,70.0)-30.0)*math.pi/180.)
    CameraSettings(camera_loc, camera_rot,'FOV',40.0) #'MILLIMETERS' # 'FOV'

else:
    camera_loc = ((random.uniform(0.0,10.0)-5.0),random.uniform(18.0,20.0),random.uniform(3,7))
    camera_rot = (random.uniform(60.0,75.0)*math.pi/180.,0.,(random.uniform(0.0,70.0)-30.0+180.0)*math.pi/180.)
    CameraSettings(camera_loc, camera_rot,'FOV',40.0) #'MILLIMETERS' # 'FOV'

####======================================Light source and Sky======================================

def Sun(loc,rot,color,intensity):
    # we have already a lamp in the scene
    #change its type to SUN
    bpy.data.lamps[0].type  = 'SUN'
    bpy.data.lamps[0].name = 'Sun'
    bpy.data.objects['Lamp'].name = 'Sun'

    ## light source extrinsic parameters
    bpy.data.objects['Sun'].location = loc
    bpy.data.objects['Sun'].rotation_euler = rot

    ## light source intrinsic paramters (color spectrum and intensity)
    bpy.data.lamps['Sun'].use_nodes = True
    bpy.data.lamps['Sun'].node_tree.nodes['Emission'].inputs['Color'].default_value = color
    bpy.data.lamps['Sun'].node_tree.nodes['Emission'].inputs['Strength'].default_value = intensity


###---------------------------------Sun with clear sky---------------------------------
def DaytimeClearSky():
    # sun rotates between -85 and 85(degrees) ~ 1.48(radians) until not visible anymore
    # sun is on a random place in the sky
    sunRot_x = random.randint(0,300)/100 - 1.5
    sunRot_y = random.randint(0,300)/100 - 1.5

    # the lower value decides the intensity and color of the sun
    temp = max(abs(sunRot_x), abs(sunRot_y))
    # normalize temp
    temp = 1-(temp/1.5)

    # the higher temp, the lower the sun's position, so the higher is the red proportion of the sun
    sunColor = (254/255,91/255,53/255)
    sunColor = tuple(temp*(1-x)+x for x in sunColor) + (1.0,)
    # create sun
    Sun((0.0,0.0,5.0),(sunRot_x, sunRot_y, 0.0),sunColor,4)

    # create clear sky
    world = bpy.data.worlds['World']
    world.use_nodes = True
    sky_tex = world.node_tree.nodes.new(type='ShaderNodeTexSky')
    world.node_tree.links.new(sky_tex.outputs['Color'], world.node_tree.nodes['Background'].inputs['Color'])



# randomly decide what weather we have today
weather = [DaytimeClearSky()]

weather[random.randint(0,len(weather)-1)]



##########---------------------Random Clouds-----------------
### RANDOM CLOUDS
# TODO: Wetterverhältnisse (Haze?)

# noise = world.node_tree.nodes.new(type='ShaderNodeTexNoise')

# ramp = world.node_tree.nodes.new(type='ShaderNodeValToRGB')

# mix = world.node_tree.nodes.new(type='ShaderNodeMixRGB')

# world.node_tree.links.new(sky_tex.outputs['Color'], world.node_tree.nodes['Mix'].inputs['Color1'])

# world.node_tree.links.new(world.node_tree.nodes['Noise Texture'].outputs['Color'], ramp.inputs['Fac'])

# world.node_tree.links.new(ramp.outputs['Color'], world.node_tree.nodes['Mix'].inputs['Color2'])

# world.node_tree.links.new(world.node_tree.nodes['Mix'].outputs['Color'], world.node_tree.nodes['Background'].inputs['Color'])

# bpy.data.worlds['World'].node_tree.nodes['ColorRamp'].color_ramp.elements[0].position = 0.3

########----------------------------Ground plane----------------------
# TODO: Himmel

def Plane(name,radius,loc,rotation,scale):
    bpy.ops.mesh.primitive_plane_add(radius=radius, location=loc)
    bpy.data.objects['Plane'].name = name

    bpy.ops.transform.rotate(value=rotation[0], axis=rotation[1], constraint_axis=(False, False, False), constraint_orientation='GLOBAL', mirror=False, proportional='DISABLED', proportional_edit_falloff='SMOOTH', proportional_size=1)
    bpy.data.objects[name].scale = scale #bpy.ops.transform.resize(value=scale)

Plane('Ground',ground_radius,(0,0,0),[0.0,(0,0,0)],[1.0,1.0,1.0])

def MaterialColor(mesh,name,albedo,roughness):
    bpy.ops.material.new()
    bpy.data.materials['Material'].name = name

    # link the material to mesh
    if mesh!=0:
        bpy.data.objects[mesh].active_material = bpy.data.materials[name]
    bpy.data.materials[name].use_nodes = True

    # now you can play with BRDF
    bpy.data.materials[name].node_tree.nodes['Diffuse BSDF'].inputs['Color'].default_value = albedo
    bpy.data.materials[name].node_tree.nodes['Diffuse BSDF'].inputs['Roughness'].default_value = roughness

#MaterialColor('Ground','ground_mat',(0.0,1.0,0.0,1.0),0.1)

def MaterialTexture(mesh,path,picName,matName,scaling):
    bpy.ops.material.new()
    bpy.data.materials['Material'].name = matName
    bpy.data.materials[matName].use_nodes = True
    bpy.data.objects[mesh].active_material = bpy.data.materials[matName]

    ## now add a texture map
    bpy.ops.image.open(filepath = path+picName)

    # rename
    tree = bpy.data.materials[matName].node_tree
    links = tree.links

    # lets add an image texture node
    img_node = tree.nodes.new(type='ShaderNodeTexImage')
    img_node.image = bpy.data.images[picName]
    # rescale image
    img_node.texture_mapping.scale = (scaling, scaling, 1)

    if 'leaf' in matName:
        # add a Mix Shader to compute alpha value and connect nodes
        mixShader = tree.nodes.new(type='ShaderNodeMixShader')
        mixShader.name = "mix_shader"
        links.new(img_node.outputs['Alpha'], mixShader.inputs[0])
        links.new(tree.nodes['Diffuse BSDF'].outputs['BSDF'], mixShader.inputs[2])
        links.new(mixShader.outputs[0], tree.nodes['Material Output'].inputs['Surface'])

        mixRGB = tree.nodes.new(type="ShaderNodeMixRGB")
        links.new(img_node.outputs['Color'], mixRGB.inputs[1])
        links.new(mixRGB.outputs[0], tree.nodes['Diffuse BSDF'].inputs['Color'])

        mixRGB.inputs[0].default_value = random.randint(70,90)/100
        mixRGB.inputs[2].default_value = foliage_color_palette[random.randint(0,len(foliage_color_palette)-1)]

        # add transparent BSDF shader
        transpBSDF = tree.nodes.new(type='ShaderNodeBsdfTransparent')
        transpBSDF.name = "transp_BSDF"
        links.new(transpBSDF.outputs[0], mixShader.inputs[1])
    else:
        links.new(img_node.outputs['Color'], tree.nodes['Diffuse BSDF'].inputs['Color'])

    ### select road as active object
    bpy.context.scene.objects.active = bpy.data.objects[mesh]
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.select_all(action = 'SELECT')
    bpy.ops.uv.smart_project() # automatically unwrap object
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')

MaterialTexture('Ground',data_location+'/textures/','gras_04.jpg','ground_mat', 1)
# MaterialTexture('Ground',data_location+'/textures/','texture_sand.jpg','ground_mat', 10)


Plane('Road',1,(0.0,0.0,0.001),[0.0,(0,0,0)],[ground_radius,1.0,1.0])
MaterialTexture('Road',data_location+'/textures/','road.jpg','road_mat', 1)

#####--------------------Fence-------------------
Plane('FenceSide'+str(1),ground_radius,(0.0,ground_radius,0),[math.pi/2.0,(1,0,0)],(1.0, 0.05, 0.1))
MaterialColor(0,'fence_mat',(71/255, 58/255, 77/255 ,1.0),0.1)
bpy.data.objects['FenceSide'+str(1)].active_material = bpy.data.materials['fence_mat']
Plane('FenceSide'+str(2),ground_radius,(0.0,-ground_radius,0),[math.pi/2.0,(1,0,0)],(1.0, 0.05, 0.1))
bpy.data.objects['FenceSide'+str(2)].active_material = bpy.data.materials['fence_mat']
Plane('FenceSide'+str(3),ground_radius,(ground_radius,0.0,0),[math.pi/2.0,(0,1,0)],(0.05, 1.0, 0.1))
bpy.data.objects['FenceSide'+str(3)].active_material = bpy.data.materials['fence_mat']
Plane('FenceSide'+str(4),ground_radius,(-ground_radius,0.0,0),[math.pi/2.0,(0,1,0)],(0.05, 1.0, 0.1))
bpy.data.objects['FenceSide'+str(4)].active_material = bpy.data.materials['fence_mat']

###---------------------------Trees---------------
MaterialColor(0,'tree_mat',(0.8, 0.168422, 0.0101812, 1),0.1)
# MaterialColor(0,'leaves_mat',(0.0, 1.0, 0.0, 1),0.1)

# foliage color palette
# foliage_color_palette = [(168/255, 24/255, 0, 1.0), (72/255, 0, 0, 1.0), (96/255, 24/255, 0, 1.0), (209/255, 76/255, 3/255, 1.0)]
foliage_color_palette = [(168/255, 24/255, 0, 1.0), (72/255, 0, 0, 1.0), (96/255, 24/255, 0, 1.0), (209/255, 76/255, 3/255, 1.0), (0.854902, 0.647059, 0.12549, 1.0), (0.4, 0.392157, 0.1, 1.0)]

tree_pos=[]
tree_sigma=[]
leafTypeNum=10
tree_leaftype=[]
tree_trunkDiam = []
tree_treeHeight = []
tree_numBranches = []
tree_numFallenLeaves = []
# set mean number of fallen leaves for scene
mean_leafFallingrate = np.random.randint(100, 300)

# set variance and covariance of leaf distribution globally for all Trees
varWind = np.random.uniform(2,5)
covWind = np.random.uniform(-3,3)

tree_counter=0
for x in [0, 10]:#np.arange(5,25,10):
    for y in [10]: #np.arange(5,25,10):
        # TODO: Verteilung der x,y Position
        trunkDiam = np.random.normal(1.0,0.1)
        treeHeight = random.randint(3,6)
        numBranches = random.randint(40,50)
        bpy.ops.curve.tree_add(do_update=True, chooseSet=str(random.randint(0,5)), bevel=True, prune=False,
                               showLeaves=True, useArm=False, seed=0, handleType='1', levels=random.randint(2,4), length=(1, 0.3, 0.6, 0.45),
                               lengthV=(0, 0, 0, 0), branches=(0, numBranches, random.randint(30,35), 10), curveRes=(3, 5, 3, 1),
                               curve=(0, -40, -40, 0), curveV=(20, 50, 75, 0), curveBack=(0, 0, 0, 0), baseSplits=0, segSplits=(0, 0, 0, 0),
                               splitAngle=(0, 0, 0, 0), splitAngleV=(0, 0, 0, 0), scale=treeHeight, scaleV=3, attractUp=0.5,
                               shape=str(random.randint(1,7)), baseSize=random.uniform(0.2,0.4), ratio=random.uniform(0.02,0.03),
                               taper=(1, 1, 1, 1), ratioPower=random.uniform(1.2,1.4), downAngle=(90, 60, 45, 45),
                               downAngleV=(0, -50, 10, 10), rotate=(140, 140, 140, 77), rotateV=(0, 0, 0, 0), scale0=trunkDiam,
                               scaleV0=0.2, pruneWidth=random.uniform(0.4,0.6), pruneWidthPeak=0.6, prunePowerHigh=0.5,
                               prunePowerLow=0.001, pruneRatio=1, leaves=random.randint(25,30), leafScale=0.17, leafScaleX=1,
                               leafDist='4', bend=random.uniform(0.0,0.2), bevelRes=0, resU=4, frameRate=1, windSpeed=2, windGust=0,
                               armAnim=False, startCurv=0)

        # save individual height, diameter and branch number parameters
        tree_trunkDiam.append(trunkDiam)
        tree_treeHeight.append(treeHeight)
        tree_numBranches.append(numBranches)

        # set number of fallen leaves for individual tree
        tree_numFallenLeaves.append(mean_leafFallingrate + np.random.randint(-mean_leafFallingrate*0.3, mean_leafFallingrate*0.3))

        # set leaf color in Trees individually, by choosing one color from list of predefined rgb values
        # make the leaf color in trees lighter in comparison to leaves on ground by adding small value to each rgb value
        tree_leafRGB = np.add(foliage_color_palette[np.random.randint(0,len(foliage_color_palette)-1)], [0.15, 0.15, 0.15, 0])
        MaterialColor(0,'leaves_mat',tree_leafRGB,0.1)

        tree_pos.append(np.array([x,y]))
        tree_sigma.append(np.array([[varWind,covWind],[-covWind,varWind]]))
        leafTypeNum = np.random.randint(1,5)
        tree_leaftype.append(leafTypeNum)

        if tree_counter==0:
            bpy.data.objects['tree'].location = [x,y,0]
            bpy.data.objects['tree'].active_material = bpy.data.materials['tree_mat']
            bpy.data.objects['leaves'].active_material=bpy.data.materials['leaves_mat']

        else:
            num='.00'+str(tree_counter) if tree_counter+1<10 else '.0'+str(tree_counter)
            bpy.data.objects['tree'+num].location = [x,y,0]
            bpy.data.objects['tree'+num].active_material = bpy.data.materials['tree_mat']
            bpy.data.objects['leaves'+num].active_material=bpy.data.materials['leaves_mat']

        tree_counter+=1


###-----------------------------Leaves----------------
## a) shape=a primitive geometric form
def leafGeom(index,location, rotation, scale, albedo):
    leafVerts = [Vector((0, 0, 0)), Vector((0.5, 0, 1/3)), Vector((0.5, 0, 2/3)), Vector((0, 0, 1)), Vector((-0.5, 0, 2/3)), Vector((-0.5, 0, 1/3))]
    leafEdges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 3]]
    leafFaces = [[0, 1, 2, 3], [0, 3, 4, 5]]

    leafMesh = bpy.data.meshes.new('Leaf'+index)
    leafObj = bpy.data.objects.new('Leaf'+index, leafMesh)
    bpy.context.scene.objects.link(leafObj)
    leafMesh.from_pydata(leafVerts, (), leafFaces)

    bpy.data.objects['Leaf'+index].location = location
    bpy.data.objects['Leaf'+index].scale = scale
    bpy.data.objects['Leaf'+index].rotation_euler = rotation

    MaterialColor('Leaf'+index,'leaf'+index+'_mat',albedo,0.1)

##b) shape=leafsnap software
def leaf(index, leaf_radius, leaf_texture, leaf_location):
    bpy.ops.mesh.primitive_grid_add(radius=leaf_radius, enter_editmode=True,location=leaf_location)
    bpy.ops.transform.vertex_random(offset=np.random.uniform(0.0,np.random.uniform(0.03,0.1)), uniform=0, normal=0, seed=5)
    bpy.ops.object.mode_set(mode = 'OBJECT')
    bpy.data.objects['Grid'].name = 'Leaf'+index
    bpy.data.objects['Leaf'+index].rotation_euler=(0,0,np.random.uniform(0.0,360.0))

    # MaterialTexture('Leaf'+index,data_location+'/textures/',leaf_texture,'leaf'+index+'_texture',leafRGB)
    MaterialTexture('Leaf'+index,data_location+'/textures/',leaf_texture,'leaf'+index+'_texture', 1)

###-----------------------------Leaf locations----------------------

def truncNormal(mean, sigmaMat, treeDiam):
    [x, y] = mean
    # make sure sampled values not in tree trunk
    while np.sqrt(np.power(x - mean[0],2) + np.power(y - mean[1],2)) < (float(treeDiam)/2):
        [x, y] = np.random.multivariate_normal(mean, sigmaMat)
    return [x, y]

leaf_counter=1
for treeIDX in range(0,len(tree_pos)):
    # sample position of leaves
    for leafIDX in np.arange(0, tree_numFallenLeaves[treeIDX]):
        [x, y] = truncNormal(tree_pos[treeIDX], tree_sigma[treeIDX], tree_trunkDiam[treeIDX])

        # only if leaf location is on property create a leaf there
        if ((-ground_radius <= x <= ground_radius) and (-ground_radius <= y <= ground_radius)):
            leaf(str(leaf_counter), np.random.normal(0.15,0.05), 'leaf_0' +str(tree_leaftype[treeIDX]+1)+ '.png', [x,y,0.02])
            leaf_counter+=1

####----------------------Import CADs
### CADs folder

folder = data_location+'/cads/'
building = 'building_17'
ext = '.blend'
for item in [building]:
    section = '\\Object\\'
    directory = folder+item+ext+section
    filename = item
    bpy.ops.wm.append(filename=filename, directory=directory)

## arrange the objects in scene layout
bpy.data.objects[building].location = [-2.33, -2.27,0.0]
bpy.data.objects[building].scale = [0.26,0.26,0.26]

###########------------------------Duplicating objects--------------------
#bpy.ops.object.select_all(action='DESELECT')
#bpy.data.objects[car].select = True
#for i in np.arange(4):
#x1 = np.random.uniform(low=0, high=6)
#y1 = np.random.uniform(low=-1, high=1)
#bpy.ops.object.duplicate(linked=True)
#ex = '.00'+str(i+1) if i+1<10 else '.0'+str(i+1)
#bpy.data.objects[car+ex].location = (x1,y1,0)

def render(filepath, frames=1, samples=10):
    bpy.data.scenes['Scene'].frame_end =frames
    bpy.data.scenes['Scene'].render.filepath = filepath
    bpy.data.scenes['Scene'].cycles.samples = samples
    bpy.ops.render.render(animation=True)

def MaskNode(name,index,objName,node_tree, OBJECTS):
	### a mask node for car objects
	idmask_leaf = node_tree.nodes.new(type='CompositorNodeIDMask')
	idmask_leaf.name = 'idmask'+name
	idmask_leaf.index = index
	node_tree.links.new(node_tree.nodes['Render Layers'].outputs['IndexOB'], idmask_leaf.inputs['ID value'])
	## multiply node for color code
	multiply_node_leaf = node_tree.nodes.new(type='CompositorNodeMixRGB')
	multiply_node_leaf.blend_type = 'MULTIPLY'
	multiply_node_leaf.inputs[1].default_value = OBJECTS[objName]
	node_tree.links.new(idmask_leaf.outputs['Alpha'], multiply_node_leaf.inputs[2])
	return multiply_node_leaf

###-------------------Annotate--------------------
def SemanticAnnotations():
    ## color coding of annotations..
    OBJECTS = {
    'FOLIAGE' : (0/255.0, 0/255.0, 255/255.0, 1.0),
    'GROUND' : (255/255.0, 0/255.0, 0/255.0, 1.0)
    }

    ## use passindex as object identifiers
    for obj in bpy.data.objects:
        if obj.name[0] == 'L':
            obj.pass_index = 1
        if obj.name[0] == 'G':
            obj.pass_index = 2

    scene = bpy.data.scenes['Scene']
    scene.use_nodes = True
    scene.render.layers['RenderLayer'].use_pass_object_index = True
    node_tree = scene.node_tree

    #add mask nodes
    multiply_node_leaf=MaskNode('_leaf',1,'FOLIAGE',node_tree, OBJECTS)
    multiply_node_ground=MaskNode('_ground',2,'GROUND',node_tree, OBJECTS)

    #if one has more than one mask node
    add_node = node_tree.nodes.new(type='CompositorNodeMixRGB')
    add_node.blend_type = 'ADD'
    node_tree.links.new(multiply_node_leaf.outputs['Image'], add_node.inputs[1])
    node_tree.links.new(multiply_node_ground.outputs['Image'], add_node.inputs[2])

    ## add a file output node
    GT_node = node_tree.nodes.new(type='CompositorNodeOutputFile')
    GT_node.base_path = outputpath+'/'
    GT_node.file_slots[0].path='im' + sys.argv[len(sys.argv)-1] + '(2)_anno'
    node_tree.links.new(add_node.outputs['Image'], GT_node.inputs['Image'])

    render(filepath=outputpath+'/im'+sys.argv[len(sys.argv)-1] + '(1)', frames=1, samples=10)

    #set new camera params
    # print(camera_rot)
    # tmp = list(camera_rot)
    # tmp[0] += 5
    # camera_rot = tuple(tmp)
    # CameraSettings(camera_loc, camera_rot,'FOV',40.0)
    bpy.data.objects['Camera'].location.x += 3
    if(bpy.data.objects['Camera'].location.y > 2):
        bpy.data.objects['Camera'].rotation_euler[2] += (-10.0)*math.pi/180
    else:
        bpy.data.objects['Camera'].rotation_euler[2] += (10.0)*math.pi/180
    ## same different angle
    # TODO: 
    # GT_node = node_tree.nodes.new(type='CompositorNodeOutputFile')
    # GT_node.base_path = outputpath+'/'
    # GT_node.file_slots.new("set2")
    # GT_node.file_slots[1].path='im_annoO'+sys.argv[len(sys.argv)-1]
    # node_tree.links.new(add_node.outputs['Image'], GT_node.inputs['Image'])

    render(filepath=outputpath+'/im'+sys.argv[len(sys.argv)-1] + '(2)', frames=1, samples=10)

    
SemanticAnnotations()
#render(filepath='scripts/output/init'+sys.argv[7], frames=1, samples=10)