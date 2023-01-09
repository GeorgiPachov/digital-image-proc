# OCR with Neural Network
The aim of this project is to demonstrate capability of an OCR application with the use a photo processing pipeline, built on OpenCV and a Neural Network, built on Tensorflow.

# Contents
1. Task definition
2. Photo processing pipeline
3. Training data gathering process
4. Neural Network
6. Data augmentation
7. Results and limitations
7. Future work and improvements
8. References


## Task definition
The goal of the project is to implement a practical computer vision application, which is able to process photos of text and recognize the characters. 
![Alt text](readme_goal.png "Goal")

## Photo processing pipeline
In order to achieve that, a robust photo processing pipeline is needed. The pipeline itself needs to be relatively robust towards:
- Different lighting conditions - some photos can be taken at daylight, others - with artificial light.
- Different focus conditions - some parts of the photo may be "crisp", others - not so much.
- "Noise" present on the paper photographed - spots, dots, crumples or other simple artifacts should not prevent the pipeline from properly analysing the image.
- Different words - pipeline should be able to distinguish between different words.
- Multiple lines - pipeline should be able to handle multiple lines and multiple words within each line. Line and word order should be preserved after being analysed.
- Different sizes of text on every line - it is not guaranteed that all text will be of the same size.

In real-world production-grade OCR systems, additional requirements are also present: angle handling, filtering of additional items present on the sheet of paper, (e.g. pen), and many many others.   
These will not be a focus of the current version of the project, but might be improved upon in the future.  

  
![Alt text](readme_pipeline.png "Pipeline")
The pipeline has 9 distinct stages:
1. Read image and convert to grayscale
2. Apply [Laplacian differential operator](https://en.wikipedia.org/wiki/Laplace_operator) on image in order to detect edges (edges are very detectable in handwriting performed with a pen)
3. Since outputs of laplacian filter performed on an image of a white sheet of paper is relatively predictable and robust, they can act as an input to a binarizing algorithm. While previously several different adaptive binarizers were attempted (including [Otsu's](https://en.wikipedia.org/wiki/Otsu%27s_method) binarization method, custom adaptive-binarization, etc), results tended to be unstable - a perfectly normal image would have had parts of it improperly binarized because of simple light and shade shifts. Using binarization after performing laplacian filter proved a lot more robust to different lightings and focus conditions.
4. After binarization, due to usually pen in handrwriting not being perfectly continuous, small "gaps" in letter definions usually appear - this is because of the edge detection algorithm does not highlight edges where handwriting is more "bleak" (less hand pressure is applied or pen does not distribute an even flow of ink). In order to deal with that, adaptive thickening algorithm was implemented. On the binarized image, bounded dilatation is applied - with number of iterations as parameter (from 1 to 3), number of found connected components are counted. The correct number of dilatation iteration is the one that minimizes the number of found components, and is between 1 and 3. The reasing behind the algorithm is as follows - at least 1 dilatation needs to be performed because of the laplacian. If by performing dilatations, number of found components decrease, that means dilatation is "connecting" components. A certain amount of "connections" is required because of the laplacian. It needs to be bounded, otherwise it would connect letters themselves. Heuristically 1 to 3 px of dilatation has proved to be effective.
Example 1:
![Alt text](readme_pipeline2.png "Pipeline")
Example 2:
![Alt text](readme_pipeline3.png "Pipeline")
5. After proper amount of "thickening" has been chosen, the next step is to locate the connected components (letters). OpenCV's connectedComponents function (https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html) proved an excellent fit for job, able to return boundaries, areas and exact pixels belonging to each connected component (letter)
6. The area for each component returned is especially important, since we are able to perform noise suppression checks based it. While in real-world application noise suppression parameters might need to be adaptive based on the photo, a simple pixel based constant proved to work good on all test samples.
![Alt text](readme_pipeline4.png "Pipeline")
7. Bounding boxes of components are extracted, cropped and saved in memory. These are the letters that will be analysed. Their positions on the photo will be used to form lines and words.
8. Lines forming. A recursive "elimination" algorithm is performed - letter bounding boxes are sorted by y, and the top one is taken. A virtual horizontal line is then created through the center of it - every letter that this line crosses is considered part of the same row. All letters on the same row (including the source one) are removed from the set and procedure is repeated until no letter is left. While being simple and fast to implement, the algorithm has a drawback that if curved handwriting is applied, it won't be detected as a single row. In future works, arcs instead of lines can be used to trace curved handwriting.
 ![Alt text](readme_pipeline5.png "Pipeline")
9. Words forming. A simple statistics-based algorithm is performed here - for each line, analyse the spaces between letters. If there is enough variation (maximum is bigger than 2 times the mean), multiple words exist - a reduced maximum (*0.95) is taken as threshold (to take into account off-by-1 pixel-based errors on word spacing columns). Spacing distributions with very little variance points out that the variance is due to the letters themselves and just a single word is present.
 ![Alt text](readme_pipeline6.png "Pipeline")

## Training data gathering process
A selection of system-available fonts was chosen - half of them machine fonts, other half 'handwritten-like'. Fonts needed to be carefully selected, as unreliable fonts exist - they have very unexpected drawings of characters and providing very little visual resemblence to the character they are supposed to represent.
```commandline
# Selection of fonts - part of them are handwritten, others are machine fonts
handwritten_fonts = ['Herculanum', 'Annai MN', 'Bradley Hand', 'Brush Script MT', 'Chalkboard',
             'Comic Sans MS', 'Luminary', 'Noteworthy', 'Papyrus', 'Party LET', 'Savoye LET', 
             'Sign Painter', 'Skia', 'Snell Roundhand', 'Times New Roman', 'Trattatello']
typed_fonts = ['Arial', 'Arial Black', 'Arial Narrow', 'Arial Rounded MT Bold', 'Copperplate', 'Courier New', 'Helvetica',
              'Impact', 'Lucinda Grande', 'Microsoft Sans Serif', 'Tahoma', 'Verdana', 'Menlo', 'Didot', 'Copperplate', 'Avenir', 'Futura']

fonts = handwritten_fonts + typed_fonts 
```

After that, for each font and for each character class (33 selected fonts, 26 character classes for english), the letter was drawn on an offscreen canvas using the specified font and saved to disk to form the initial dataset.
This is how the data for class (character) "A" looks like:
 ![Alt text](readme_data1.png "Sample of class A")

A random sample of the data generated for the neural network is demonstrated below. Take a note of the very different styles of letters present in the system-available fonts:
 ![Alt text](readme_data2.png "Random data sample")

## Convolutional Neural Network
 ![Alt text](readme_nn1.png "Conv NN")

A very small convolutional neural network was used - a total of 8 layers - 3 convolutional, 2 max pooling 1 flatten and 2 dense layers.
The network is purposefuly chosen to be small. Character bounding boxes were expected to be small and the distinction of letters is a relatively simple visual task. So not a lot of capacity was needed for the network. 
Tensorflow was chosen as framework to implement the network on.   
While papers do exist for similar visual tasks citing use of AlexNet, LeNet, etc, that would require entail significantly more training and testing data and better hardware to train on.

## Data augmentation
The initial train set contained 33*26 = 858 images. Trained on it, the neural network was able to produce 85% accuracy on the test set.
Additional data augmentation was performed to improve performance.   
Since in handwriting slight incline is common, rotations on -15, -10, -5, 5, 10, 15 degrees were performed for each image.
Additionally, erosion and dilatation (single iteration) were performed on the images (this gave the nice benefit of recognizing text written with marker instead of pen).
For each image in the dataset, additional 18 versions were added (6 rotations x {Dilatation, No Change, Erosion}).
The resulting dataset consisted of 15444 images.

Random sample of the augmented data:
 ![Alt text](readme_augmented1.png "Conv NN")

## Results and limitations
The test set consists of 16 images of sheets with text - total of 162 characters.
Most of them were written with pen, a few of them - with marker.
The network achieved peak accuracy of 97% with early-stopping tuning on the test set.
While additional validation set must be used to further fine-tune the network, due do various time pressures - this was not done in this case. Also, for the same reason, no other fine-tuning methods were attempted - no tuning of learning rates, batch sizes and even model architecture were performed.
A more broad and diverse test set needs to be gathered before making definitive statements about accuracy. 

## Future work and improvements
Several improvements can be made:
- More diverse test set can be collected - containing handwriting of multiple people, phographed from different angles, with different cameras/phones, etc.
- Additional existing datasets can be used - e.g. EMNIST
- That would allow for usage of a bigger network with more capacity
- Pipeline can be improved significantly - smarter binarization can be found (or learned - e.g. a binarization CNN can be trained to extract the text from a photo). This would exclude the need for using edge detection filters and would improve pipeline stability and robustness.
- After higher levels of accuracy are achieved, dictionary-based correction module can be applied - correcting simple errors of the OCR program.

## References


