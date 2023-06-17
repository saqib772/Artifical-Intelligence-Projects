### Cartoonify Image

This Python script allows you to transform an input image into a cartoon-like version using the OpenCV library. It provides a graphical user interface (GUI) to select an image file and applies various image processing techniques to achieve the cartoon effect.

## Prerequisites
Before running the script, make sure you have the following dependencies installed:

- cv2 (OpenCV): for image processing
- easygui: to open the file dialog
- numpy: to store the image
- imageio: to read the image stored at a particular path
- sys
- matplotlib: for plotting images
- os
- tkinter: for creating the GUI
- PIL: for working with images

  ## How to Use
1. Run the script.
2. The GUI window will appear.
3. Click on the "Cartoonify an Image" button to select an image file.
4. The selected image will be displayed along with a series of transformations.
5. Adjustments will be made to the grayscale image, followed by smoothening and edge retrieval using thresholding techniques.
6. The script will then prepare a mask image and apply a bilateral filter to enhance the cartoon effect.
7. Finally, the cartoon image will be displayed, and you can choose to save it by clicking on the "Save cartoon image" button.

Please note that the script will automatically save the cartoon image in the same directory as the original image, with the filename "CartoonImage" followed by the original image's extension.

Enjoy transforming your images into captivating cartoons! 😄
