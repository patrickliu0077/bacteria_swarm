Swarm Analyzer
This is a Python script that analyzes images of bacterial swarms to extract quantitative data such as the number of swarms, their sizes, shapes, and morphological features like the number of "fingers" (protrusions). The script processes the image, identifies swarms, calculates various metrics, and outputs the results both visually and in a CSV file.

Table of Contents
Features
Requirements
Installation
Usage
Parameters
Example
Output
How It Works
1. Loading the Image
2. Preprocessing
3. Contour Detection
4. Analyzing Swarms
5. Visualization
6. Saving Data
Customization
Troubleshooting
License
Features
Automated Swarm Detection: Identifies bacterial swarms in an image using image processing techniques.
Quantitative Analysis: Calculates metrics such as area, perimeter, circularity, center coordinates, radius, and the number of "fingers" (protrusions) for each swarm.
Visualization: Generates and displays annotated images with detected swarms highlighted.
Data Export: Saves the analyzed data into a CSV file for further analysis.
Customizable Parameters: Allows users to adjust thresholds and parameters to suit different images and conditions.
Requirements
Python 3.x
Libraries:
OpenCV (opencv-python)
NumPy
Matplotlib
Pandas
Sys (standard library)
Installation
Clone the Repository or Download the Script:

bash
Copy code
git clone https://github.com/yourusername/swarm-analyzer.git
cd swarm-analyzer
Install Required Libraries:

Use pip to install the necessary Python libraries:

bash
Copy code
pip install opencv-python numpy matplotlib pandas
If you're using Anaconda, you can create a new environment and install the packages:

bash
Copy code
conda create -n swarm-analyzer python=3.8
conda activate swarm-analyzer
conda install -c conda-forge opencv numpy matplotlib pandas
Usage
Prepare Your Image:

Ensure that your image is in a format supported by OpenCV (e.g., JPEG, PNG).
Place the image in a known directory and note its path.
Update the Script:

Open the script file in a text editor.

Update the image_path parameter in the SwarmAnalyzer instantiation to point to your image.

python
Copy code
analyzer = SwarmAnalyzer(
    image_path='/path/to/your/image.jpg',  # Replace with your image path
    # ... other parameters ...
)
Adjust Parameters (Optional):

Modify parameters like area_threshold_min, area_threshold_max, defect_distance, etc., to suit your image.
Run the Script:

From the command line:

bash
Copy code
python swarm_analyzer.py
Or, if you're using a Jupyter notebook or interactive environment, run the cells containing the code.

Parameters
image_path (string): Path to the input image.
output_csv (string, default='swarm_data.csv'): Path to the output CSV file.
output_image (string, default='processed_image.jpg'): Path to the output annotated image.
area_threshold_min (int, default=500): Minimum area for a contour to be considered a swarm.
area_threshold_max (int or None, default=None): Maximum area for a contour to be considered a swarm.
defect_distance (int, default=500): Minimum distance for convexity defects to be considered significant (used in counting "fingers").
blur_kernel_size (tuple, default=(5, 5)): Kernel size for Gaussian blur.
threshold_value (int, default=50): Threshold value for binary thresholding (used if threshold_method is 'binary_inv').
max_value (int, default=255): Maximum value to use with the thresholding function.
threshold_method (string, default='otsu'): Thresholding method to use ('otsu', 'adaptive', or 'binary_inv').
Example
python
Copy code
analyzer = SwarmAnalyzer(
    image_path='/Users/liuyizhou/Desktop/Brandon_Swarm/swarm947.jpg',
    output_csv='swarm_data.csv',
    output_image='processed_image.jpg',
    area_threshold_min=80000,
    area_threshold_max=200000,
    defect_distance=50,
    blur_kernel_size=(5, 5),
    threshold_value=5,
    max_value=255,
    threshold_method='otsu'
)
Output
Annotated Image:

Saved to the file specified by output_image (e.g., processed_image.jpg).
Displays the detected swarms with contours and enclosing circles drawn.
CSV Data File:

Saved to the file specified by output_csv (e.g., swarm_data.csv).
Contains the following columns:
center_x, center_y: Coordinates of the swarm's center.
radius: Radius of the minimum enclosing circle.
fingers: Number of convexity defects (protrusions) detected.
area: Area of the swarm contour.
perimeter: Perimeter of the swarm contour.
circularity: Circularity metric of the swarm.
Console Output:

Displays the DataFrame containing the swarm data.
Prints the number of contours found (optional).
Visualization:

Shows the thresholded image and the annotated processed image during execution.
How It Works
The script performs several steps to analyze the swarms:

1. Loading the Image
The image is loaded using OpenCV's cv2.imread function.
The script checks if the image was loaded successfully; otherwise, it raises a FileNotFoundError.
2. Preprocessing
Grayscale Conversion: The image is converted to grayscale to simplify processing.
Gaussian Blur: A Gaussian blur is applied to reduce noise and smooth the image.
Thresholding: The blurred image is thresholded to create a binary image where swarms are highlighted against the background.
Methods:
Otsu's Thresholding ('otsu'): Automatically calculates the optimal threshold value.
Adaptive Thresholding ('adaptive'): Calculates thresholds for small regions, useful for uneven lighting.
Binary Inverse Thresholding ('binary_inv'): Uses a fixed threshold value.
Visualization: The thresholded image can be displayed for inspection.
3. Contour Detection
Contours are detected in the binary image using cv2.findContours.
The retrieval mode cv2.RETR_TREE is used to retrieve all contours and reconstruct a full hierarchy.
Contours are stored for further analysis.
4. Analyzing Swarms
Sorting Contours: Contours are sorted by area in descending order.
Excluding the Petri Dish: The largest contour (assumed to be the petri dish) is excluded from analysis.
Filtering Contours:
Contours are filtered based on area_threshold_min and area_threshold_max.
This step removes noise and ensures only swarms are analyzed.
Calculating Metrics:
Minimum Enclosing Circle: Finds the smallest circle that can enclose the contour.
Convex Hull and Defects: Used to identify and count the number of "fingers" or protrusions.
Area and Perimeter: Calculated using OpenCV functions.
Circularity: Calculated using the formula 
Circularity
=
4
𝜋
×
Area
Perimeter
2
Circularity= 
Perimeter 
2
 
4π×Area
​
 .
Data Collection: The calculated metrics are stored in a list of dictionaries for each swarm.
Visualization:
Contours and circles are drawn on the image for each detected swarm.
5. Visualization
Annotated Image:
The processed image with annotations is converted from BGR to RGB color space for display.
The image is displayed using Matplotlib.
Saving the Image:
The annotated image is saved to the specified output file.
6. Saving Data
DataFrame Creation:
The swarm data is converted into a Pandas DataFrame.
Console Output:
The DataFrame is printed to the console.
CSV Export:
The DataFrame is saved as a CSV file for further analysis.
Customization
You can adjust several parameters to optimize the analysis for your images:

Thresholding Parameters:
Change threshold_method to 'adaptive' or 'binary_inv' if Otsu's method doesn't yield good results.
Adjust threshold_value if using 'binary_inv'.
Area Thresholds:
Modify area_threshold_min and area_threshold_max to include all swarms and exclude noise or unwanted objects (like the petri dish).
Defect Distance:
Adjust defect_distance to fine-tune the detection of "fingers" based on the morphology of your swarms.
Blur Kernel Size:
Change blur_kernel_size to control the amount of smoothing. A larger kernel size reduces more noise but may blur important details.
Visualization:
Uncomment or comment the Matplotlib visualization code in the preprocess_image and visualize methods to enable or disable image displays during execution.
Troubleshooting
No Swarms Detected:
Ensure that area_threshold_min is set low enough to include your swarms.
Verify that the image is correctly loaded and that the path is accurate.
Too Many Objects Detected (Noise):
Increase area_threshold_min to exclude smaller noise contours.
Adjust the thresholding parameters to improve segmentation.
Petri Dish Detected as Swarm:
Make sure the largest contour is being excluded as the petri dish.
Adjust area_threshold_max to exclude very large contours.
Incorrect Finger Counts:
Modify defect_distance to filter out insignificant convexity defects.
Visualization Windows Not Showing:
If running in a non-interactive environment, the Matplotlib windows may not display. Ensure that you are running the script in an environment that supports GUI operations.
Errors During Execution:
Check that all required libraries are installed and that there are no typos in the code.
Ensure that the image file is accessible and not corrupted.