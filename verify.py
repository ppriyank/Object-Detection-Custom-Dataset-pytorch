from datasets import PascalVOCDataset
import torch.utils.data
from utils import *
from torchvision import transforms
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import os 

import json
with open('TRAIN_images.json') as json_file:
    images = json.load(json_file)

with open('TRAIN_objects.json') as json_file:
    objects = json.load(json_file)

resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def true_box(image, boxes, labels,name="verify/", id=0):
	try:
	    os.mkdir(name)
	except OSError:
		None
	    
	
	annotated_image = image
	# image = normalize(to_tensor(resize(image)))
	det_boxes = boxes.to('cpu')
	# original_dims = torch.FloatTensor([300, 300, 300, 300]).unsqueeze(0)
	# det_boxes = det_boxes * original_dims
	det_labels = [rev_label_map[l] for l in labels.to('cpu').tolist()]
	# images_array = image.to('cpu').permute(1,2,0).numpy()
	# temp = (images_array - np.min(images_array)) * 255 // (np.max(images_array) - np.min(images_array))
	# annotated_image = Image.fromarray(np.array(temp).astype(np.uint8))
	original_image = annotated_image
	# original_image.save(name + "_temp2.jpg", "JPEG")
	draw = ImageDraw.Draw(annotated_image)
	font = ImageFont.truetype("./arial.ttf", 15)
	label_color_map
	for i in range(det_boxes.size(0)):
	    # Boxes
	    box_location = det_boxes[i].tolist()
	    draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
	    draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
	        det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
	    # Text
	    text_size = font.getsize(det_labels[i].upper())
	    text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
	    textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
	                        box_location[1]]
	    draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
	    draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
	              font=font)
	annotated_image.save(name + str(id) + ".jpg", "JPEG")


for i,x in enumerate(images):
	print(i,x)
	original_image = Image.open(x, mode='r')
	original_image = original_image.convert('RGB')
	object = objects[i]
	boxes = torch.FloatTensor(object['boxes'])  
	labels = torch.LongTensor(object['labels']) 
	true_box(original_image, boxes, labels, id=i)



