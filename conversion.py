import csv
import os  

path = os.path.abspath("/scratch/pp1953/cml/ass/class_pics/")
n_objects = 0
count =3
dict ={}
unique_labels = {'Laptop':1, 'Person':2, "background": 0}
with open('annotations.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        print(row)
        image_path = path + "/" + row[0]
        boxs = [int(x) for x in row[2:]]
        if image_path in dict:
        	dict[image_path]["boxes"].append(boxs)  
        	dict[image_path]["labels"].append( unique_labels[row[1]]  )  
        	dict[image_path]["difficulties"].append(0 )  
        else:
        	dict[image_path] = {"boxes": [boxs], "labels": [unique_labels[row[1]]], "difficulties": [0] } 
        if row[1] not in unique_labels:
        	unique_labels[row[1]] = count
        	count +=1



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



import json 
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
