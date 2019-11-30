# Object-Detection-Custom-Dataset-pytorch
Training object detection on custom dataset 


# Label your dataset
Use this tool (conveneint) https://github.com/tzutalin/labelImg  

Store the labels in "anno" folder (or any other folder, just change the path in `annotation.py`). Also change the path of images `path` in   `annotation.py` for all images.  

`temp_dict` in `annotation.py` maps the labeling tool class ids to desired training ids  
change `Original_image_height`  and `Original_image_width` 