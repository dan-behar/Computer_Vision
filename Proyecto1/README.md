## Proyecto1
This project was aimed to create a car plate detector that can recognize a plate and the letters

* model.ipynb: the jupyter notebook in which I created, trained and exported the model. The file "model.sav" was too big to be uploaded to Github, so I didn't upload it
* script.ipynb: the jupyter notebook where I developed the main program previous the .py file
detector.py: the python file that runs everyting
* For the model, I extracted the training letters from this [kaggle](https://www.kaggle.com/datasets/aladdinss/license-plate-digits-classification-dataset/data) and I stored them in a directory called **model_images**
* images85.jpg: An image to test the project

### Instructions to run it:
1. Run model.ipynb to produce the model. The kaggle dataset is needed for this step
2. In the CMD run: python detector.py --p <COMPLETE_DIRECTION_TO_FILE/IMG_NAME.IMG_TYPE> or python detector.py