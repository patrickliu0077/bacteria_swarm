import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import json

class PetriDishDetector:
    def __init__(self, image_path, root_window, output_image='detected_petri_dishes.jpg',
                 blur_kernel_size=(5, 5), dp=1.2, min_dist=100, param1=50, param2=30,
                 min_radius=50, max_radius=0):
        self.image_path = image_path
        self.root_window = root_window
        self.output_image = output_image
        self.blur_kernel_size = blur_kernel_size  # Kernel size for Gaussian blur
        self.dp = dp  # Inverse ratio of the accumulator resolution to the image resolution
        self.min_dist = min_dist  # Minimum distance between the centers of the detected circles
        self.param1 = param1  # Higher threshold for the Canny edge detector
        self.param2 = param2  # Accumulator threshold for the circle centers
        self.min_radius = min_radius  # Minimum circle radius
        self.max_radius = max_radius  # Maximum circle radius (0 means no maximum)
        self.image = None
        self.gray = None
        self.processed_image = None
        self.circles = None

    def load_image(self):
        """
        Loads the image from the specified path and resizes it for better processing.
        """
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            print(f"Error: Image at path {self.image_path} could not be loaded.")
            sys.exit(1)
        
        # Calculate new dimensions (keeping aspect ratio)
        max_dimension = 1000
        height, width = self.image.shape[:2]
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize the image
        print(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
        self.image = cv2.resize(self.image, (new_width, new_height))
        self.processed_image = self.image.copy()

    def preprocess_image(self):
        """
        Converts the image to grayscale and applies Gaussian blur.
        """
        # Convert to grayscale
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple symmetric blurs instead of one asymmetric blur
        kernel_y, kernel_x = map(int, self.blur_kernel_size)
        
        # If kernels are very different, apply them separately
        if abs(kernel_x - kernel_y) > 4:
            # First apply vertical blur
            self.gray = cv2.GaussianBlur(self.gray, (1, kernel_y), 0)
            # Then apply horizontal blur
            self.gray = cv2.GaussianBlur(self.gray, (kernel_x, 1), 0)
        else:
            # For similar kernel sizes, use single symmetric blur
            kernel_size = min(kernel_x, kernel_y)
            self.gray = cv2.GaussianBlur(self.gray, (kernel_size, kernel_size), 0)

    def detect_petri_dishes(self):
        """
        Detects circles in the image using the Hough Circle Transform.
        """
        print(f"Gray image shape: {self.gray.shape}")
        print(f"Value range: min={self.gray.min()}, max={self.gray.max()}")
        
        # Calculate radius range based on image width
        width = self.gray.shape[1]
        height = self.gray.shape[0]
        min_radius = int(width * (1/6))  # 1/3 diameter = 1/6 radius
        max_radius = int(width * (1/4))  # 1/2 diameter = 1/4 radius
        
        print(f"Using radius range: {min_radius} to {max_radius}")
        
        circles = cv2.HoughCircles(
            self.gray,
            cv2.HOUGH_GRADIENT,
            dp=1.5,
            minDist=min_radius * 2,  # Ensure circles don't overlap
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        # Filter circles that are too close to the edge
        if circles is not None:
            circles = np.uint16(np.around(circles))
            valid_circles = []
            
            for circle in circles[0]:
                x, y, r = circle
                # Check if circle is completely within the image bounds
                if (x >= r and x < width - r and 
                    y >= r and y < height - r):
                    valid_circles.append(circle)
            
            if valid_circles:
                self.circles = np.array([valid_circles])
                print(f"Number of valid circles detected: {len(valid_circles)}")
                print(f"Circle details:")
                for i, circle in enumerate(valid_circles):
                    print(f"Circle {i}: center=({circle[0]}, {circle[1]}), radius={circle[2]}")
            else:
                self.circles = None
                print("No valid circles found after edge filtering")
        else:
            self.circles = None
            print("No circles detected. Try adjusting parameters:")
            print(f"Current params: dp={self.dp}, minDist={min_radius * 2}, param1=50, param2=30")

    def visualize(self):
        """
        Draws the detected circles on the image and displays it in a Tkinter window.
        """
        if self.circles is not None:
            for i in self.circles[0, :]:
                # Draw only the outer circle in green
                cv2.circle(self.processed_image, (i[0], i[1]), i[2], (0, 255, 0), 2)

        # Convert the image for Tkinter
        image_rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Create a new top-level window
        window = tk.Toplevel()
        window.title('Detected Petri Dishes')
        
        # Create a label for the legend
        legend = tk.Label(window, text='Green circles indicate detected petri dishes', 
                         bg='white', pady=5)
        legend.pack(side='bottom', fill='x')
        
        # Convert to PhotoImage for Tkinter
        photo = ImageTk.PhotoImage(image=image_pil)
        
        # Create canvas to display the image
        canvas = tk.Canvas(window, width=image_pil.width, height=image_pil.height)
        canvas.pack(side='top', fill='both', expand=True)
        
        # Add image to canvas
        canvas.create_image(0, 0, anchor='nw', image=photo)
        
        # Keep a reference to avoid garbage collection
        canvas.image = photo
        
        # Save the processed image
        cv2.imwrite(self.output_image, self.processed_image)

    def mask_petri_dishes(self):
        """
        Creates a mask that includes only the petri dishes.
        Returns the masked image.
        """
        mask = np.zeros_like(self.gray)
        if self.circles is not None:
            for i in self.circles[0, :]:
                # Draw filled circles on the mask
                cv2.circle(mask, (i[0], i[1]), i[2], (255), thickness=-1)

        # Apply the mask to the original image
        masked_image = cv2.bitwise_and(self.image, self.image, mask=mask)

        # Save the masked image
        cv2.imwrite('masked_petri_dishes.jpg', masked_image)
        
        return masked_image

    def detect_swarm_regions_blue(self, square_image, kernel_size=(5,5)):
        """
        Detects the petri dish edge using connected components.
        Returns both the image with blue edges and the inner boundary mask.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(square_image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 51, 2
        )
        
        # Use the passed kernel_size parameter
        kernel = np.ones(kernel_size, np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
        
        # Filter components by area
        min_area = square_image.shape[0] * square_image.shape[1] * 0.01  # 1% of image
        valid_labels = []
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] > min_area:
                valid_labels.append(i)
        
        # Create mask of valid components
        mask = np.zeros_like(gray)
        for label in valid_labels:
            mask[labels == label] = 255
        
        # Find the contour of the combined regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = square_image.copy()
        inner_mask = np.zeros_like(gray)
        
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            # Draw with blue color
            cv2.drawContours(result, [largest_contour], -1, (255, 0, 0), 2)
            
            # Create inner boundary mask (shrunk by 50 pixels instead of 100)
            cv2.drawContours(inner_mask, [largest_contour], -1, 255, -1)
            kernel = np.ones((30,30), np.uint8)  # 50 pixels on each side (reduced from 201x201)
            inner_mask = cv2.erode(inner_mask, kernel, iterations=1)
        
        return result, inner_mask

    def detect_swarm_regions(self, square_image, inner_mask=None):
        """
        Detects swarm regions focusing on the white-opaque edge characteristic.
        """
        if inner_mask is None:
            return square_image.copy()
        
        # First apply the mask to the original image
        masked_image = cv2.bitwise_and(square_image, square_image, mask=inner_mask)
        
        # Pre-processing steps
        # 1. Denoise with reduced strength
        denoised = cv2.fastNlMeansDenoisingColored(masked_image, None, 7, 7, 5, 15)
        
        # 2. Convert to LAB and enhance contrast
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge((l,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 3. Edge-preserving bilateral filter
        bilateral = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 4. Multi-channel detection
        hsv = cv2.cvtColor(bilateral, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
        
        # HSV detection for green regions
        lower_green1 = np.array([71, 40, 50])
        upper_green1 = np.array([84, 125, 255])
        
        # Create intensity mask for white-opaque edges
        _, intensity_mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        
        # Combine green and intensity detection
        green_mask = cv2.inRange(hsv, lower_green1, upper_green1)
        
        # Create gradient mask to detect edges
        gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        gradient = np.absolute(gradient)
        gradient = np.uint8(255 * gradient / np.max(gradient))
        
        # Combine all masks with weights
        mask = cv2.bitwise_and(green_mask, gradient)
        mask = cv2.bitwise_or(mask, intensity_mask)
        
        # Apply inner mask
        mask = cv2.bitwise_and(mask, inner_mask)
        
        # 5. Morphological operations
        kernel_size = max(square_image.shape[0] // 250, 3)
        kernel_close = np.ones((kernel_size, kernel_size), np.uint8)
        kernel_open = np.ones((kernel_size//2, kernel_size//2), np.uint8)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        # 6. Contour detection with edge adherence
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result = square_image.copy()
        if contours:
            min_area = (square_image.shape[0] * square_image.shape[1]) * 0.015
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                
                # Refine contour points to stick to edges
                refined_contour = []
                for point in largest_contour:
                    x, y = point[0]
                    # Search for strong edges or white-opaque regions
                    roi_gray = gray[max(0, y-2):min(gray.shape[0], y+3),
                                  max(0, x-2):min(gray.shape[1], x+3)]
                    roi_gradient = gradient[max(0, y-2):min(gradient.shape[0], y+3),
                                         max(0, x-2):min(gradient.shape[1], x+3)]
                    
                    if roi_gray.size > 0 and roi_gradient.size > 0:
                        # Combine intensity and gradient information
                        combined_score = roi_gray.astype(float) + roi_gradient.astype(float)
                        y_offset, x_offset = np.unravel_index(np.argmax(combined_score), roi_gray.shape)
                        refined_contour.append([[x + x_offset - 2, y + y_offset - 2]])
                
                if refined_contour:
                    refined_contour = np.array(refined_contour)
                    epsilon = 0.003 * cv2.arcLength(refined_contour, True)
                    approx_contour = cv2.approxPolyDP(refined_contour, epsilon, True)
                    cv2.drawContours(result, [approx_contour], -1, (0, 0, 255), 2)
        
        return result

    def parameter_tuning_mode(self, image_idx=1):
        """
        Runs parameter tuning mode on a specific image index.
        Shows 30 different parameter combinations.
        """
        if self.circles is None or image_idx >= len(self.circles[0]):
            print("Invalid image index or no circles detected")
            return

        # Get the specified square
        circle = self.circles[0][image_idx]
        x, y, r = circle
        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(self.image.shape[1], x + r)
        y2 = min(self.image.shape[0], y + r)
        square = self.image[int(y1):int(y2), int(x1):int(x2)]

        # Generate 30 parameter combinations
        parameter_sets = []
        for i in range(30):
            params = {
                'lower_h': np.random.randint(60, 75),
                'lower_s': np.random.randint(30, 70),
                'lower_v': np.random.randint(30, 70),
                'upper_h': np.random.randint(80, 95),
                'upper_s': np.random.randint(90, 150),
                'upper_v': np.random.randint(200, 255)
            }
            parameter_sets.append(params)

        # Create subplot grid
        rows = 5
        cols = 6
        plt.figure(figsize=(20, 16))
        
        for idx, params in enumerate(parameter_sets):
            result = self.detect_swarm_regions(square, params)
            
            plt.subplot(rows, cols, idx + 1)
            plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            plt.title(f'Set {idx + 1}')
            plt.axis('off')
            
            # Print parameters for each set
            print(f"\nParameter Set {idx + 1}:")
            print(f"Lower HSV: ({params['lower_h']}, {params['lower_s']}, {params['lower_v']})")
            print(f"Upper HSV: ({params['upper_h']}, {params['upper_s']}, {params['upper_v']})")
        
        plt.tight_layout()
        plt.show()

    def collect_square_regions(self):
        """
        Collects square regions and detects swarm areas.
        """
        regions_data = []
        square_images = []
        processed_squares = []
        
        # Calculate midpoint of image width
        midpoint = self.image.shape[1] // 2
        
        # Revert kernel mapping to previous state
        kernel_mapping = {
            1: (5, 5),
            2: (5, 5),
            3: (5, 5),
            4: (7, 7),
            5: (5, 5),
            6: (7, 7)
        }
        
        if self.circles is not None:
            for idx, circle in enumerate(self.circles[0], 1):
                x, y, r = circle  # r is the detected radius
                
                # Get square region
                x1 = max(0, x - r)
                y1 = max(0, y - r)
                x2 = min(self.image.shape[1], x + r)
                y2 = min(self.image.shape[0], y + r)
                
                square = self.image[int(y1):int(y2), int(x1):int(x2)]
                
                # Get kernel size from mapping, default to (5,5) if not specified
                kernel_size = kernel_mapping.get(idx, (5, 5))
                
                # First detect petri dish edge (blue method)
                blue_result, inner_mask = self.detect_swarm_regions_blue(square, kernel_size=kernel_size)
                
                # Then detect swarm regions within the inner boundary (red method)
                square_with_swarm = self.detect_swarm_regions(square, inner_mask)
                
                # Calculate swarm area using red contours
                hsv = cv2.cvtColor(square_with_swarm, cv2.COLOR_BGR2HSV)
                # Red color range in HSV
                lower_red1 = np.array([0, 50, 50])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([170, 50, 50])
                upper_red2 = np.array([180, 255, 255])
                
                # Create masks for both red ranges
                mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                red_mask = cv2.bitwise_or(mask1, mask2)
                
                # Find contours in the red mask
                contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                swarm_area = 0
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    swarm_area = cv2.contourArea(largest_contour)
                
                square_images.append(square)
                processed_squares.append(square_with_swarm)
                
                # Determine side (left or right)
                side = 'left' if x < midpoint else 'right'
                
                # Store data with kernel size and detected radius
                regions_data.append({
                    'region_id': idx,
                    'center_x': int(x),
                    'center_y': int(y),
                    'side': side,
                    'swarm_area': swarm_area,
                    'blue_kernel_size': f"{kernel_size[0]},{kernel_size[1]}",
                    'detected_radius': int(r),  # Store the detected radius
                    'validated': None  # Initialize validation field
                })
            
            # Show validation dialog
            validation_dialog = SwarmValidationDialog(self.root_window, processed_squares, 
                                                    [idx for idx in range(1, len(self.circles[0]) + 1)])
            validation_dialog.dialog.wait_window()
            
            # Add only validation results to the DataFrame
            for row in regions_data:
                region_id = row['region_id']
                row['validated'] = validation_dialog.results[region_id].get()
            
            return pd.DataFrame(regions_data), processed_squares
        
        return None, None

class ParameterDialog:
    def __init__(self, parent):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Parameter Settings")
        self.result = None
        
        # Store parameter history
        self.parameter_history = []
        self.current_params = {
            'blur_kernel': "5,5",
            'max_dimension': 1000,
            'dp': 1.5,
            'param1': 50,
            'param2': 30,
            'radius_min_factor': 1/6,
            'radius_max_factor': 1/4
        }
        self.parameter_history.append(self.current_params.copy())
        
        # Create parameter entries with current values
        self.params = {
            'blur_kernel': tk.StringVar(value=self.current_params['blur_kernel']),
            'max_dimension': tk.IntVar(value=self.current_params['max_dimension']),
            'dp': tk.DoubleVar(value=self.current_params['dp']),
            'param1': tk.IntVar(value=self.current_params['param1']),
            'param2': tk.IntVar(value=self.current_params['param2']),
            'radius_min_factor': tk.DoubleVar(value=self.current_params['radius_min_factor']),
            'radius_max_factor': tk.DoubleVar(value=self.current_params['radius_max_factor'])
        }
        
        # Create and pack widgets
        row = 0
        tk.Label(self.dialog, text="Blur Kernel (x,y):").grid(row=row, column=0)
        tk.Entry(self.dialog, textvariable=self.params['blur_kernel']).grid(row=row, column=1)
        
        row += 1
        tk.Label(self.dialog, text="Max Dimension (100-2000):").grid(row=row, column=0)
        tk.Entry(self.dialog, textvariable=self.params['max_dimension']).grid(row=row, column=1)
        
        row += 1
        tk.Label(self.dialog, text="DP (1.0-3.0):").grid(row=row, column=0)
        tk.Entry(self.dialog, textvariable=self.params['dp']).grid(row=row, column=1)
        
        row += 1
        tk.Label(self.dialog, text="Param1 (10-100):").grid(row=row, column=0)
        tk.Entry(self.dialog, textvariable=self.params['param1']).grid(row=row, column=1)
        
        row += 1
        tk.Label(self.dialog, text="Param2 (10-100):").grid(row=row, column=0)
        tk.Entry(self.dialog, textvariable=self.params['param2']).grid(row=row, column=1)
        
        row += 1
        tk.Label(self.dialog, text="Min Radius Factor (0.1-0.3):").grid(row=row, column=0)
        tk.Entry(self.dialog, textvariable=self.params['radius_min_factor']).grid(row=row, column=1)
        
        row += 1
        tk.Label(self.dialog, text="Max Radius Factor (0.2-0.5):").grid(row=row, column=0)
        tk.Entry(self.dialog, textvariable=self.params['radius_max_factor']).grid(row=row, column=1)
        
        # Add output folder structure options
        row += 1
        tk.Label(self.dialog, text="\nOutput Folder Options:").grid(row=row, column=0, columnspan=2)
        
        row += 1
        self.create_subfolders = tk.BooleanVar(value=True)
        tk.Checkbutton(self.dialog, text="Create organized subfolders", 
                      variable=self.create_subfolders).grid(row=row, column=0, columnspan=2)
        
        # Buttons frame
        row += 1
        button_frame = tk.Frame(self.dialog)
        button_frame.grid(row=row, column=0, columnspan=2, pady=10)
        
        tk.Button(button_frame, text="OK", command=self.validate_and_save).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Revert to Previous", command=self.revert_to_previous).pack(side=tk.LEFT, padx=5)
        
        # Save parameters to file button
        tk.Button(button_frame, text="Save Parameters", command=self.save_parameters_to_file).pack(side=tk.LEFT, padx=5)
        
        # Load parameters from file button
        tk.Button(button_frame, text="Load Parameters", command=self.load_parameters_from_file).pack(side=tk.LEFT, padx=5)
        
    def revert_to_previous(self):
        """Revert to the previous parameter set"""
        if len(self.parameter_history) > 1:
            # Remove current parameters
            self.parameter_history.pop()
            # Get previous parameters
            previous_params = self.parameter_history[-1]
            
            # Update all entry fields
            for name, value in previous_params.items():
                self.params[name].set(value)
            
            messagebox.showinfo("Parameters Reverted", "Parameters have been restored to previous values.")
        else:
            messagebox.showinfo("No History", "No previous parameters available.")
    
    def save_parameters_to_file(self):
        """Save current parameters to a file"""
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
                title="Save Parameters"
            )
            if file_path:
                current_params = {name: var.get() for name, var in self.params.items()}
                with open(file_path, 'w') as f:
                    json.dump(current_params, f, indent=4)
                messagebox.showinfo("Success", "Parameters saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save parameters: {str(e)}")
    
    def load_parameters_from_file(self):
        """Load parameters from a file"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json")],
                title="Load Parameters"
            )
            if file_path:
                with open(file_path, 'r') as f:
                    loaded_params = json.load(f)
                
                # Update all entry fields
                for name, value in loaded_params.items():
                    if name in self.params:
                        self.params[name].set(value)
                
                # Add to history
                self.parameter_history.append(loaded_params)
                messagebox.showinfo("Success", "Parameters loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load parameters: {str(e)}")
            
    def validate_and_save(self):
        try:
            # Validate all parameters
            blur = [int(x) for x in self.params['blur_kernel'].get().split(',')]
            if len(blur) != 2 or not all(1 <= x <= 31 and x % 2 == 1 for x in blur):
                raise ValueError("Blur kernel must be two odd numbers between 1 and 31")
                
            max_dim = self.params['max_dimension'].get()
            if not 100 <= max_dim <= 2000:
                raise ValueError("Max dimension must be between 100 and 2000")
                
            dp = self.params['dp'].get()
            if not 1.0 <= dp <= 3.0:
                raise ValueError("DP must be between 1.0 and 3.0")
                
            # Store current parameters in history
            current_params = {name: var.get() for name, var in self.params.items()}
            self.parameter_history.append(current_params)
            
            # Store result for main program
            self.result = current_params
            self.dialog.destroy()
            
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))

class SwarmValidationDialog:
    def __init__(self, parent, processed_squares, region_ids):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Validate Swarm Regions")
        self.results = {}  # Store validation results
        self.radii = {}    # Store radii measurements
        self.photo_references = []  # Keep photo references
        
        # Create a canvas and scrollbar for potentially many images
        canvas = tk.Canvas(self.dialog)
        scrollbar = ttk.Scrollbar(self.dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # For each processed square
        for idx, square in enumerate(processed_squares):
            if square is None:  # Skip if image is None
                continue
                
            region_id = region_ids[idx]
            frame = ttk.Frame(scrollable_frame)
            frame.pack(pady=10, padx=10)
            
            try:
                # Convert OpenCV image to PhotoImage
                img_rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_pil.thumbnail((400, 400))  # Resize for display
                img_tk = ImageTk.PhotoImage(image=img_pil)
                self.photo_references.append(img_tk)  # Keep reference
                
                # Display image
                label = tk.Label(frame, image=img_tk)
                label.image = img_tk  # Keep a reference
                label.pack()
                
                # Validation buttons
                validation_var = tk.StringVar(value="Pending")
                self.results[region_id] = validation_var
                
                buttons_frame = ttk.Frame(frame)
                buttons_frame.pack(pady=5)
                
                ttk.Button(buttons_frame, text="Yes", 
                          command=lambda v=validation_var: v.set("Yes")).pack(side=tk.LEFT, padx=5)
                ttk.Button(buttons_frame, text="No", 
                          command=lambda v=validation_var: v.set("No")).pack(side=tk.LEFT, padx=5)
                
                # Display detected radius (read-only)
                radius_frame = ttk.Frame(frame)
                radius_frame.pack(pady=5)
                ttk.Label(radius_frame, text=f"Detected Radius: {self.circles[0][idx][2]} pixels").pack()
                
                # Label for region ID
                ttk.Label(frame, text=f"Region {region_id}").pack()
                
            except Exception as e:
                print(f"Error processing region {region_id}: {str(e)}")
                continue
        
        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Done button
        ttk.Button(self.dialog, text="Done", command=self.done).pack(pady=10)
        
        # Center the dialog
        self.dialog.update_idletasks()
        width = min(self.dialog.winfo_width(), self.dialog.winfo_screenwidth() - 100)
        height = min(self.dialog.winfo_height(), self.dialog.winfo_screenheight() - 100)
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    def done(self):
        self.dialog.destroy()

def create_output_folders(base_folder):
    """Creates organized folder structure for outputs"""
    folders = {
        'processed_images': os.path.join(base_folder, 'processed_images'),
        'masked_images': os.path.join(base_folder, 'masked_images'),
        'detected_regions': os.path.join(base_folder, 'detected_regions'),
        'data': os.path.join(base_folder, 'data')
    }
    
    # Create each folder if it doesn't exist
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
        
    return folders

# Modify main execution
# Modify main execution
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Show parameter dialog by default
    param_dialog = ParameterDialog(root)
    root.wait_window(param_dialog.dialog)
    if param_dialog.result is None:
        sys.exit("Parameter selection cancelled")
    params = param_dialog.result

    # File selection
    folder_path = filedialog.askdirectory(title="Select folder containing images")
    if not folder_path:
        sys.exit("No folder selected")

    # Create output folders if requested
    output_folders = None
    if params.get('create_subfolders', True):  # Default to True if not specified
        output_folders = create_output_folders(folder_path)

    # Process images with selected parameters
    all_results = []
    
    # Define specific blur kernel sizes for each image
    blur_kernels = {
        1: (5, 5),
        3: (5, 5),
        4: (7, 7),
        6: (7, 7),
        2: (7, 7),
        5: (7, 7)
    }
    
    jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]

    for idx, image_file in enumerate(jpg_files, start=1):
        image_path = os.path.join(folder_path, image_file)
        
        # Select the blur kernel based on the specified image index
        selected_blur_kernel = blur_kernels.get(idx, (5, 5))  # Default to (5, 5) if not specified
        
        # Create detector instance with the selected blur kernel
        detector = PetriDishDetector(
            image_path,
            root_window=root,  # Pass the root window
            blur_kernel_size=selected_blur_kernel,
            dp=params['dp'],
            param1=params['param1'],
            param2=params['param2']
        )
        
        # Process image
        detector.load_image()
        detector.preprocess_image()
        detector.detect_petri_dishes()

        # If output folders exist, save results in organized structure
        if output_folders:
            # Save processed image
            detector.output_image = os.path.join(
                output_folders['processed_images'],
                f'processed_{image_file}'
            )
            detector.visualize()

            # Save masked image
            masked_image = detector.mask_petri_dishes()
            cv2.imwrite(
                os.path.join(output_folders['masked_images'], f'masked_{image_file}'),
                masked_image
            )

            # Process and save individual regions
            results_df, processed_squares = detector.collect_square_regions()

            if results_df is not None:
                # Save data
                results_df.to_csv(
                    os.path.join(output_folders['data'], f'results_{image_file}.csv'),
                    index=False
                )

                # Save individual region images
                for idx, square in enumerate(processed_squares, 1):
                    cv2.imwrite(
                        os.path.join(output_folders['detected_regions'], 
                                   f'region_{idx}_{image_file}'),
                        square
                    )

                all_results.append(results_df)
        else:
            # Basic processing without organized output
            detector.visualize()
            detector.mask_petri_dishes()
            results_df, _ = detector.collect_square_regions()
            if results_df is not None:
                all_results.append(results_df)

    # Combine all results if any exist
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        if output_folders:
            final_df.to_csv(
                os.path.join(output_folders['data'], 'combined_results.csv'),
                index=False
            )
        print("\nFinal Combined Results:")
        print(final_df)

    root.destroy()
