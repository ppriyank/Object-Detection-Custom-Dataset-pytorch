import glob
import os  
import json 
import xml.etree.ElementTree as ET


unique_labels = {'laptop':1, 'face':2, 'light':3, "drinks":4, "projector":5, "background": 0}

path = os.path.abspath("/scratch/pp1953/cml/ass/class_pics/")
annotation_path = "/scratch/pp1953/cml/ass/anotations/"
dict ={}

for file in glob.glob(annotation_path + "*.xml"): 
  print(file)
  temp = file.split("/")[-1][:-3]
  if temp == 'classes.':continue
  image_name= temp + "jpg"
  image_path = path + "/" + image_name
  print(image_path)
  tree = ET.parse(file)
  root = tree.getroot()
  boxs =[] 
  labels =[]
  difficulties = []
  for object in root.iter('object'):
    label = object.find('name').text.lower().strip()
    bbox = object.find('bndbox')
    xmin = int(bbox.find('xmin').text) - 1
    ymin = int(bbox.find('ymin').text) - 1
    xmax = int(bbox.find('xmax').text) - 1
    ymax = int(bbox.find('ymax').text) - 1
    boxs.append([xmin, ymin, xmax, ymax])
    labels.append(unique_labels[ label  ])
    difficulties.append(0)
  dict[image_path] = {"boxes": boxs, "labels": labels, "difficulties": difficulties} 

   			


train_images = list()
train_objects = list()
test_images = list()
test_objects = list()

for key in dict:
	name = key.split("/")[-1]
	if name !=  'IMG_0504.jpg' and name != 'IMG_0505.jpg':
		train_images.append(key)
		train_objects.append(dict[key])
	else:
		test_images.append(key)
		test_objects.append(dict[key])




output_folder = './'
# Save to file
with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
    json.dump(train_images, j)


with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
    json.dump(train_objects, j)


with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
    json.dump(unique_labels, j)  # save label map too


with open(os.path.join(output_folder, 'TEST_images.json'), 'w') as j:
    json.dump(test_images, j)


with open(os.path.join(output_folder, 'TEST_objects.json'), 'w') as j:
    json.dump(test_objects, j)
