## Proyecto1
This project was aimed to create a car plate detector that can recognize a plate and the letters

* detector.py: the python file that runs everyting
* images85.jpg: An image to test the project
* model.ipynb: the jupyter notebook in which I created, trained and exported the model. The file "model.sav" was too big to be uploaded to Github, so I didn't upload it
  * For the model, I extracted the training letters from this [kaggle](https://www.kaggle.com/datasets/aladdinss/license-plate-digits-classification-dataset/data) and I stored them in a directory called **model_images**
* model.py the python file that creates the model that detector will use
* script.ipynb: the jupyter notebook where I developed the main program previous the .py file

### Instructions to run it:
1. Download the [kaggle](https://www.kaggle.com/datasets/aladdinss/license-plate-digits-classification-dataset/data) dataset
2. In model.py add in the DIR VARIABLE the direction to reach the folder
3. Run model.py to produce the model
4. In the CMD run: python detector.py --p <COMPLETE_DIRECTION_TO_FILE/IMG_NAME.IMG_TYPE> or just python detector.py
