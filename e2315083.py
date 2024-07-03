import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

 
#1 a) 

def createPath():
    script_dir = os.path.dirname('dataset')  # I got the directory of the current script
    
    # I initialize input and output path for original_1.jpeg
    imgPathforOriginal_1 = os.path.join(script_dir, 'original_1.jpeg')
    scaledPath_1 = os.path.join(script_dir, 'scaled_1.jpeg')
    
    # I called scale_image function for oiriginal_1.jpeg, and I gave scale factor = 3
    scale_image(imgPathforOriginal_1, scaledPath_1, 3)

    # I initialize input and output path for original_2.jpeg
    imgPathforOriginal_2 = os.path.join(script_dir, 'original_2.jpeg')
    scaledPath_2 = os.path.join(script_dir, 'scaled_2.jpeg')

    # I called scale_image function for oiriginal_1.jpeg, and I gave scale factor = 3

    scale_image(imgPathforOriginal_2, scaledPath_2, 3)

#1 b)

def scale_image(input_path, output_path, scale_factor):
    # I read the image from specified input_path in grayScale mode.
    img = cv.imread(input_path, cv.IMREAD_GRAYSCALE)
    # I extracted height and width of the image with using img.shape
    # image.shape returns the dimensions of image array.
    height, width = img.shape[:2]
    # I resized image with resize function. The new dimensions of the images are 
    # calculated by multiplying original dimensions with scale_factor.
    img_resize = cv.resize(img, (width * scale_factor, height * scale_factor))

    # Save the files 
    cv.imwrite(output_path, img_resize)


#1 c)

def rotateImage(input_path, output_path):
    # I read the image from specified input_path in grayScale mode.
    img = cv.imread(input_path, cv.IMREAD_GRAYSCALE) 
    # I extracted height and width of the image with using img.shape
    # image.shape returns the dimensions of image array.
    height, width = img.shape[:2]
    # In order to rotate 45 degree, I created rotation matrix
    # This rotationmatrix function needs three things: the center of rotation
    # degree, and scale factor which is 1 because we dont want to scale image
    # in this step.
    T = cv.getRotationMatrix2D((width/2, height/2), 45, 1)  

    #I used cv.warpAffine to transform the image using the specified matrix
    Img_Transform = cv.warpAffine(img, T, (width, height))
    cv.imwrite(output_path, Img_Transform)
    cv.waitKey(0)


#2 b)

def SIFT(input_path):
    # I read the image from specified input_path in grayScale mode.
    img = cv.imread(input_path, cv.IMREAD_GRAYSCALE) 
    
    #In order to detect key points I created sift objects
    sift = cv.SIFT_create()
    
    # sift.detect allow us to find key points 'None' means no mask is used, so keypoints
    # are detected throughout the entire image
    mykeypoints = sift.detect(img,None)
    # In order to draw keypoints of the image I used cv.drawkeypoints. It takes
    # original image,keypoints and output of the images, and flag argument specify 
    # that keypoints are drawn with size and orientation. 
    img = cv.drawKeypoints(img,mykeypoints,img,flags = cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.figure()
    plt.imshow(img)
    plt.show()


#2 c), d), e)

def BruteForce(input_path_1,input_path_2):

    img1 = cv.imread(input_path_1,cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(input_path_2,cv.IMREAD_GRAYSCALE)
    # In order to initialize SIFT detector, I used SIFT_create.
    sift = cv.SIFT_create()
    # I used sift.detectAndCompute to find the key point and their descriptor vector.
    # The key point is a point of the image, and the descriptor is a vector which 
    # describes the local surrounding  around that point.
    key1,descriptor1 = sift.detectAndCompute(img1,None)
    key2,descriptor2 = sift.detectAndCompute(img2,None)

    
    # I created Brute Force Matcher object to match descriptors between two images
    bf = cv.BFMatcher()


    # In order to matches descriptors from the first image to the second, I used bf.match function
    # bf.matches comparing a descriptor from one set to every descriptor in the other set to find matches.
    # It computes distances betweeen descriptors and considers closest as most likely match (Euclidean distance L2)

    matches = bf.match(descriptor1,descriptor2)
    # I sorted matches based on distance to distinguish better matches
    matches = sorted(matches, key=lambda x: x.distance)
    
    # I drawed the top ten matches between two images
    imgMatch = cv.drawMatches(img1,key1,img2,key2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Matches returned match objects, and eatch match objects contains attributes that supploy
    # info about the match.
    # queryIdx attribute refers to index of the first set of descriptor matched with a descriptor
    # in the second set. In below, I collected the indices of the key-points .
    myIndices = [match.queryIdx for match in matches]

    # distance attribute gives the distance between the matched descriptors. This list 
    # provides distance from each match     
    myDistances = [match.distance for match in matches]

    # Plotting figure

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(imgMatch)
    ax1.set_title('Matching Keypoints')
    ax1.axis('off')

    ax2.plot(myIndices, myDistances, marker='o', linestyle='None')
    ax2.set_xlabel('Indices of Key Points')
    ax2.set_ylabel('Matching Distance')
    ax2.set_title('Matching Distance versus Key Point Indices')
    ax2.grid(True)

    plt.show()



   #f)
    # When an image is enlarged up, its distinctive characteristics (such as edges and corners) get bigger.
    # Some feature descriptors designed to be a scale-invariant. For example, in this
    # assignment such as SIFT can modify their scale parameter to match related features at different scales.
    # As a result, even if one image is resized, the matching distance between the same features remains low (demonstrating good matches).
    # If an image scale down, features decrease and may become less noticeable. However, comparing these properties may be limited, particularly at highly tiny scales when detail loss occurs.


    #g)
 
    #As a picture rotates, the positional and angular connections between its points alter. 
    #In this assignment SIFT which is rotatet-invariant.
    #Rotatet-invariant descriptors change the orientation of the keypoints to ensure that the descriptor 
    #remains consistent throughout rotations. As a result, the matching distance remains reasonably stable 
    #and does not increase dramatically with rotation angle. 

    #Reducing the rotation angle returns the features to their original orientation, 
    #which should reduce the matching distance if previous rotations had influenced it. 
    #The matching distance between features should be small across all rotation degrees
    #in fully rotation-invariant systems.




if __name__ == '__main__':


    script_dir = os.path.dirname('dataset')

    imgPath_1 = os.path.join(script_dir, 'original_1.jpeg')
    imgPath_2 = os.path.join(script_dir, 'original_2.jpeg')



    # Output paths for rotated images
    rotatedPath_1 = os.path.join(script_dir, 'rotated_1.jpeg')
    rotatedPath_2 = os.path.join(script_dir, 'rotated_2.jpeg')

    # Scale both images
    createPath()  
    scaledPath_1 = os.path.join(script_dir,'scaled_1.jpeg')
    scaledPath_2 = os.path.join(script_dir,'scaled_2.jpeg')
    # Rotate the first image
    rotateImage(imgPath_1, rotatedPath_1)
    # Rotate the second image
    rotateImage(imgPath_2, rotatedPath_2)
    # Generate SIFT features and plot
    SIFT(imgPath_1)     #SIFT for orginal
    SIFT(imgPath_2)
    SIFT(scaledPath_1)
    SIFT(scaledPath_2)  # SIFT for scaled
    SIFT(rotatedPath_1)
    SIFT(rotatedPath_2) #SIFT for rotated.
    
    # Feature matching
    BruteForce(imgPath_1,scaledPath_1)
    BruteForce(imgPath_1,rotatedPath_1)
    BruteForce(imgPath_2,scaledPath_2)
    BruteForce(imgPath_2,rotatedPath_2)



    
   


