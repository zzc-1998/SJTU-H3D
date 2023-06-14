import numpy as np
import trimesh
import pandas as pd
import os 
import time 
import argparse,pprint,csv

def mesh_or_scene(obj):
    try:
        # only trimesh object has vertices attribute. if the obj contains multiple meshes, it is loaded as scene obj.
        vertices = obj.vertices
        return obj
    except:
        print("The object is a scene with mulitple meshes, the main mesh is used.")
        meshes = list(obj.geometry.values())
        max = 0
        for i in range(len(meshes)):
            vertices = len(meshes[i].vertices)
            if vertices > max:
                max = vertices
                max_i = i
        return list(obj.geometry.values())[max_i]
        
    

def get_geometry_loss(objpath):
  #print("Begin loading mesh.")   
  mesh = trimesh.load(objpath)
  mesh = mesh_or_scene(mesh)
  # extract geometry features
  dihedral_angle = mesh.face_adjacency_angles
  return [np.mean(dihedral_angle)]

def main(config):
    print('----------------------------------------------------------------')
    print('Begin geometry_loss_quality_measure calculation')
    print('----------------------------------------------------------------')
    headers = ['objs','dihedral_angle_mean']
    data = [headers]
    objs = pd.read_csv(config.info_path)['Image']
    for obj in objs:
        start = time.time()
        obj_path = os.path.join(config.data_path, obj, obj+ '.obj')
        print(obj_path)
        data.append([obj_path] + get_geometry_loss(obj_path))
        end = time.time()
        print('Time cost ' + str(end-start) + 's.')

    with open(config.output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='ddh')
    parser.add_argument('--info_path', type=str, default='path_to_datainfo')
    parser.add_argument('--data_path', type=str, default='path_to_projections')
    parser.add_argument('--output_csv', type=str, default='path_to_output_csv')

    config = parser.parse_args()
    pprint.pprint(config.__dict__)
    main(config)
