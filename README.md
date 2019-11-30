# Object-Detection-Custom-Dataset-pytorch
Training object detection on custom dataset 


# Label your dataset
Use this tool (conveneint) https://github.com/tzutalin/labelImg  
Make sure you save it in PASCAL (.xml file) format  

# Convert into trainable format
Store the labels in "anotations" folder (or any other folder, just change the path in `annotation.py`). Also change the path of images `path` in   `annotation.py` for all images.  

```python annotation.py```

Creates "TEST_images.json" , "TEST_objects.json" , "TRAIN_images.json" , "TRAIN_objects.json"  

format: `Train/Test_images.json` : [list of files names]  
format: `Train/Test_objects.json` : [{"boxes": [[..], [...], [...]], "labels": "boxes": [[..], [...], [...]]} , {...}, {...} ... ]    


# Verification

```python verify.py```



