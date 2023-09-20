import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
  
def imgview(image):
    """Receives and array and shows the image sent on the array
    Args: 
        image (np array): Source image
    Returns:
        plot: the image shown
    """
    k=13
    figure = plt.figure(figsize=(k,k))
    axis1 = figure.add_subplot(2,2,1)

    if (len(image.shape)>2):
        axis1.imshow(image)
    else:
        axis1.imshow(image,cmap='gray',vmin=0,vmax=255)

    plt.axis('off')
    plt.plot()

def hist(image):
    """Receives an image's np array and prints the image and its histogram
    Args: 
        imgage (np array): Source image
    Returns:
        plot: the image and the histogram in the same frame
    """
    k=8
    fig = plt.figure(figsize=(k,k))
    
    axis1 = fig.add_subplot(2,2,1)
    axis1.imshow(image,cmap='gray',vmin=0,vmax=255)
    plt.axis('off')

    axis2 = fig.add_subplot(2,2,2)
    axis2.set_title("Histogram")
    axis2.set_xlabel("Pixel Count")
    axis2.set_ylabel("Pixel Vount")
    axis2.set_facecolor("black")
    histr = cv.calcHist([image],[0],None,[256],[0,256])
    axis2.plot(histr, c="white",linewidth=0.6)

    plt.plot()

def colorhist(img):
    fig, ax = plt.subplots(figsize=(20,8))
    colors = ['r','g','b']
    for i, color in enumerate(colors):
        histr = cv.calcHist([img],[i],None,[256],[0,256])
        ax.plot(histr, c=color, alpha=0.9)
        x = np.arange(0.0, 256, 1)
    ax.set_xlim([0,256])
    ax.grid(alpha=0.2)
    ax.set_facecolor('k')
    ax.set_title('Histogram', fontsize=20)
    ax.set_xlabel('Pixel value', fontsize=20)
    ax.set_ylabel('Pixel count', fontsize=20)

    plt.show()


def imgcmp(image1, image2, titles = None):
    """Receives to images, each on its array and shows both images to allow comparison between them
    Args: 
        image1 (np array): Source image number 1
        image2 (np array): Source image number 2
    Returns:
        plot: the images shown
    """
    if titles == None:
        k=8
        fig = plt.figure(figsize=(k,k))
        
        axis1 = fig.add_subplot(2,2,1)
        plt.axis('off')
        axis2 = fig.add_subplot(2,2,2)
        plt.axis('off')

        if (len(image1.shape)>2):
            axis1.imshow(image1)
            plt.axis('off')
            axis2.imshow(image2)
            plt.axis('off')
        else:
            axis1.imshow(image1,cmap='gray',vmin=0,vmax=255)
            axis2.imshow(image2,cmap='gray',vmin=0,vmax=255)
        
        plt.plot()
    else: 
        if len(titles) < 2:
            return 0

        k=8
        fig = plt.figure(figsize=(k,k))
        
        axis1 = fig.add_subplot(2,2,1)
        axis1.set_title(titles[0])
        plt.axis('off')
        ax2 = fig.add_subplot(2,2,2)
        ax2.set_title(titles[1])
        plt.axis('off')

        if (len(image1.shape)>2):
            axis1.imshow(image1)
            plt.axis('off')
            axis2.imshow(image2)
            plt.axis('off')
        else:
            axis1.imshow(image1,cmap='gray',vmin=0,vmax=255)
            axis2.imshow(image2,cmap='gray',vmin=0,vmax=255)
        
        plt.plot()


def imgcdf(img):
    """Compute the CDF on an image
    Args: 
        img (numpy array): Source image
    Returns:
        cdf (list): Computed CDf of img
        hist (list): Histogram of img
    """
    hist_list = cv.calcHist([img],[0],None,[256],[0,256])
    hist = hist_list.ravel()

    cdf = []
    t = 0
    for p in hist:
        t += p
        cdf.append(t)
    return cdf, hist

def imgeq(img):
    """ Equalize a grayscale image
    Args:
        img (numpy array): Grayscale image to equalize
    Returns:
        eq (numpy array): Equalized image
    """
    cdf = imgcdf(img)[0]
    cdf_eq = []
    n = img.shape[0] * img.shape[1]
    m = min(i for i in cdf if i > 0)

    for i in cdf:
        if i >= m:
            cdf_eq.append(int(round(255*(i-m)/(n-m))))
        else:
            cdf_eq.append(0)
    eq = cv.LUT(img, np.array(cdf_eq).astype(np.uint8))
    return eq