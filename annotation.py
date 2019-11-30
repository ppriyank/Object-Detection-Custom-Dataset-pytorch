import glob
import os  
import json 

unique_labels = {'Laptop':1, 'Person':2, 'Lights':3, "Drinks":4, "Projector":5, "background": 0}
temp_dict ={15 : 'Lights', 16: 'Person' , 17: 'Laptop', 18: "Drinks", 19: "Projector"}

path = os.path.abspath("/scratch/pp1953/cml/ass/class_pics/")
annotation_path = "/scratch/pp1953/cml/ass/anno/"
dict ={}

Original_image_height = 3024
Original_image_width = 4032

for file in glob.glob(annotation_path + "*.txt"): 
	print(file)	
	temp = file.split("/")[-1][:-3]
	if temp == 'classes.': continue
	image_name= temp + "jpg"
	image_path = path + "/" + image_name
	filepath = 'Iliad.txt'
	with open(file) as fp:
   		line = fp.readline()
   		while line:
   			string = line.strip()
   			print("Line : {}".format(string))
   			row = string.split(" ")
   			line = fp.readline()
   			boxs =[] 
   			for i,x in enumerate(row[1:]):
   				x = float(x)
   				if i%2 == 0:
   					boxs.append(int(x* Original_image_width))
   				else:
   					boxs.append(int(x* Original_image_height))
   			if image_path in dict:
   				dict[image_path]["boxes"].append(boxs)
   				dict[image_path]["labels"].append( unique_labels[  temp_dict[ int(row[0])]  ]  )
   				dict[image_path]["difficulties"].append(0 )
   			else:
   				dict[image_path] = {"boxes": [boxs], "labels": [unique_labels[  temp_dict[ int(row[0])]]], "difficulties": [0] } 

   			


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
