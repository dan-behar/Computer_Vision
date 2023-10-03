import numpy as np
import cv2 as cv
import joblib
import sys

def extent(cnt):
    """ Receives the dims of a contour and returns its values
        
    Args:
        cnt (np array): The contour

    Returns:
        res (np array): The dimentions
    """

    area = cv.contourArea(cnt)
    x,y,w,h = cv.boundingRect(cnt)
    rect_area = w*h
    res = float(area)/rect_area
    return res

def cortador(img_raw):
    """ Receives the direction of the image to process and returns the plate
        
    Args:
        img_raw (str): The direction of the image

    Returns:
        img_raw (np array): The array of the image with the plate marked
        objetive (np array): The array of the plate section
        xp (int): The X position of the starting point of the box marking the plate
        yp (int): The Y position of the starting point of the box marking the plate
    """

    # Variances of each color. I'm searching for the lowest variance
    red = np.var(img_raw[:,:,0])
    green = np.var(img_raw[:,:,1])
    blue = np.var(img_raw[:,:,2])

    if blue < green and blue < red:
        imgray = img_raw[:,:,2]
    elif green < blue and green < red:
        imgray = img_raw[:,:,1]
    else:
        imgray = img_raw[:,:,0]

    # Normalization of the image
    _, imgray = cv.threshold(imgray, 210, 255,cv.THRESH_TRUNC)
    _, imgray = cv.threshold(imgray, 60, 255,cv.THRESH_TOZERO)
    imgray = cv.GaussianBlur(imgray, (5,5), 10)

    # Getting all the contours
    thresh = cv.adaptiveThreshold(imgray, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,17,1)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

    # Getting the plate searching the most rectangular one
    img = img_raw.copy() 
    m, n = 0, 0
    for i, contorn in enumerate(contours):
        ext = extent(contorn)
        ext = round(ext, 2)
        x,y,w,h = cv.boundingRect(contorn)
        img = cv.drawContours(img,[contorn],0,(0,255,0),2)
        img = cv.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        if (ext > m):
            n = i 
            m = ext

    # Plate identified
    x, y, w, h = cv.boundingRect(contours[n])
    xp, yp = x, y
    objective = img_raw[y:y+h, x:x+w]
    cv.rectangle(img_raw, (x, y), (x + w, y + h), (0, 255, 0), 4)
    print("ETAPA 1 FINALIZADA")

    return img_raw, objective, xp, yp

def components(objective):
    """ Receives the plate in order to extract all the components
        
    Args:
        objetive (np array): The array of the plate section

    Returns:
        img, plate
        components (array): The array with necessary information to get the letters
        min_height (int): The value of the min height that we will use to search for a letter. This related to the whole size of the plate
        max_height (int): The value of the max height that we will use to search for a letter. This related to the whole size of the plate
        hist (np array): The histogram of the ammount of black and white in the objetive image
        mvalue (int): The mean value of the plate
        img (np array): The array of the plate section but in color
        plate (np array): The array of the plate section but in gray and normalized
    """
    # Getting the selected part
    h, w, _ = objective.shape
    c_w = int(0.04 * w)
    c_h = int(0.11 * h)
    x1 = c_w
    y1 = c_h
    x2 = w - c_w
    y2 = h - c_h
    img = objective[y1:y2, x1:x2]

    # Getting the original picture with the new coordinates
    plate = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    plate = cv.GaussianBlur(plate, (7,7), 1)
    hist = cv.calcHist([plate], [0], None, [256], [0, 256])

    # Binarizing
    glow = sum(hist[100:])
    threshold = 0.5
    if glow / sum(hist) > threshold:
        img_bin = cv.adaptiveThreshold(plate, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 33, 5)
    else:
        img_bin = cv.adaptiveThreshold(plate, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 23, -2)

    # Cleaning the new image
    kernel = np.ones((3, 3), np.uint8)
    img_bin = cv.erode(img_bin, kernel, iterations=1)
    components = cv.connectedComponentsWithStats(img_bin, connectivity=4)

    # Getting the contours of all the elements
    contours = []
    w_med = []
    h_med = []

    for i in range(1, components[0]):
        x, y, w, h, area = components[2][i]
        contours.append(area)
        
    mindex = np.argmax(contours)
    mvalue = contours[mindex] * 0.10

    # Getting all the contours in the cutted image
    for i in range(1, components[0]):
        x, y, w, h, area = components[2][i]
        if area >= mvalue:
            if h > w:
                w_med.append(w)
                h_med.append(h)

    if len(h_med) != 0:
        max_height = (sum(h_med)/len(h_med)) * 1.40
        min_height = (sum(h_med)/len(h_med)) * 0.75
    else:
        max_height = plate.shape[0]
        min_height = 0
    print("ETAPA 2 FINALIZADA")

    return components, min_height, max_height, hist, mvalue, img, plate

def letras(components, min_height, max_height, hist, mvalue, img, img_raw, plate):
    """ Receives the plate in order to extract all the components
        
    Args:
        img, plate
        components (array): The array with necessary information to get the letters
        min_height (int): The value of the min height that we will use to search for a letter. This related to the whole size of the plate
        max_height (int): The value of the max height that we will use to search for a letter. This related to the whole size of the plate
        hist (np array): The histogram of the ammount of black and white in the objetive image
        mvalue (int): The mean value of the plate
        img (np array): The array of the plate section but in color
        img_raw (np array): The array of the whole picture with the plate marked with a rectangle
        plate (np array): The array of the plate section but in gray and normalized
    
    Retuns:
        entregable (np array): img_raw picture with the letters of the plate in the image
        placa (str): The string of the plate's letters
    """

    # Putting the letters in black and everything else in white
    glow = sum(hist[100:])
    threshold = 0.5
    if glow / sum(hist) > threshold:
        img_bin2 = cv.adaptiveThreshold(plate, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 33, 5)
    else:
        img_bin2 = cv.adaptiveThreshold(plate, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 23, -2)

    # Getting the position of the letters
    letters = []
    for i in range(1, components[0]):
        x, y, w, h, area = components[2][i]
        if area >= mvalue:
            if h > w:
                if h > min_height and h < max_height:
                    letters.append([x, y, w, h, area])

    # Ordering the letters
    letters_ordered = []
    temp = []

    for i in range(len(letters)+1):
        if i == len(letters):
            sorter = lambda x: (x[0])
            temp = sorted(temp, key=sorter)
            for j in range(len(temp)):
                letters_ordered.append(temp[j])
        else:
            if len(temp) == 0:
                temp.append(letters[i])
            else:
                if (letters[i][1] - letters[i-1][1]) < 10:
                    temp.append(letters[i])
                else:
                    sorter = lambda x: (x[0])
                    temp = sorted(temp, key=sorter)
                    for j in range(len(temp)):
                        letters_ordered.append(temp[j])
                    temp = []
                    temp.append(letters[i])

    # Removing elements that are not letters but the cutter told that they are
    ared = []
    ablue = []
    agreen = []
    for i in range(len(letters_ordered)):
        x, y, w, h, area = letters_ordered[i]
        red = np.var(img[y:y+h, x:x+w][:,:,0])
        green = np.var(img[y:y+h, x:x+w][:,:,1])
        blue = np.var(img[y:y+h, x:x+w][:,:,2])
        if blue < green and blue < red:
            ablue.append(letters_ordered[i])
        elif green < blue and green < red:
            agreen.append(letters_ordered[i])
        else:
            ared.append(letters_ordered[i])

    if len(ablue) == 1:
        letters_ordered.remove(ablue[0])
    if len(ared) == 1:
        letters_ordered.remove(ared[0])
    if len(agreen) == 1:
        letters_ordered.remove(agreen[0])

    # Saving the letters for identification
    letters = []
    for i in range(len(letters_ordered)):
        x, y, w, h, area = letters_ordered[i]
        if area >= mvalue:
            if h > w:
                if h > min_height and h < max_height:
                    letra = img_bin2[y:y+h, x:x+w]
                    letra = np.pad(letra, 5, 'constant', constant_values=255)
                    new_size = cv.resize(letra, dsize=(75, 100), interpolation=cv.INTER_LANCZOS4)
                    flatted = new_size.flatten()
                    letters.append(flatted)

    # Using the model to make the prediction
    model = joblib.load("model.sav")
    placa = ""

    letra = model.predict(letters)
    for i in range(len(letra)):
        placa = placa + letra[i]

    font = cv.FONT_HERSHEY_SIMPLEX
    org = (xp,yp - 5)
    fontScale = 1
    color = (0, 255, 0)
    thickness = 2

    entregable = cv.putText(img_raw, placa, org, font, 
                    fontScale, color, thickness, cv.LINE_AA)
    
    return entregable, placa

# Main executable
try:
    if len(sys.argv) < 2:
        car = 'images85.jpg'
    elif sys.argv[1] == "--p":
        car = sys.argv[2]
    else:
        raise Exception(f"{sys.argv[1]} not recognized")

except:
    raise Exception("Unexpected error occured")

raw = cv.imread(car, cv.IMREAD_COLOR)
raw = cv.cvtColor(raw, cv.COLOR_BGR2RGB)
hole_image, plate, xp, yp = cortador(raw)

parts, minh, maxh, historgram, mval, img, preplaca = components(plate)

placa, texto = letras(parts, minh, maxh, historgram, mval, img, raw, preplaca)

cv.imshow(f'Placa {texto}',placa)
cv.waitKey(0)
print(f"La placa es: {texto}")