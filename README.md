# Object-Detection-Custom-Dataset-pytorch (SSD 300)
Training object detection on custom dataset 

Credits: [github](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/tree/0d38943b68e2664039e4c22b2838d634b656a927)

# Label your dataset
Use this tool (conveneint) https://github.com/tzutalin/labelImg  
Make sure you save it in PASCAL (.xml file) format  
`voc_labels` in `utils.py` according to your new labels. 


# Convert into trainable format
Store the labels in "anotations" folder (or any other folder, just change the path in `annotation.py`). Also change the path of images `path` in   `annotation.py` for all images.  

```python annotation.py```

Creates "TEST_images.json" , "TEST_objects.json" , "TRAIN_images.json" , "TRAIN_objects.json"  

format: `Train/Test_images.json` : [list of files names]  
format: `Train/Test_objects.json` : [{"boxes": [[..], [...], [...]], "labels": "boxes": [[..], [...], [...]]} , {...}, {...} ... ]    


# Verification

```python verify.py``` (check `verify/` folder)  

**Something like This**   
   <img src="https://github.com/ppriyank/Object-Detection-Custom-Dataset-pytorch/blob/master/verify/7.jpg" width="700">



# Training 

```python modified_train.py``` (check `verify/`)   

if you want to load pretrained weights : [gdrive](https://drive.google.com/file/d/19pgfEaFIUSiQ9bcwqBbU0bTDmk2KK0cJ/view?usp=sharing) 
