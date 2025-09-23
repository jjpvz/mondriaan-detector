import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math

class GeometricImageAnalyzer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = None
        self.processed_image = None
        self.lines = []
        self.boxes = []
        
    def load_image(self):
        """Load the image and prepare it for processing"""
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {self.image_path}")
        
        # Convert to RGB for matplotlib display
        self.original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        print(f"Image loaded: {self.original_image.shape}")
        
    def preprocess_image(self):
        """Preprocess image for line and box detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        self.processed_image = edges
        return edges
    
    def detect_lines(self):
        """Detect lines using Hough Line Transform"""
        if self.processed_image is None:
            self.preprocess_image()
            
        # Use HoughLinesP for better line detection
        lines = cv2.HoughLinesP(
            self.processed_image,
            rho=1,
            theta=np.pi/180,
            threshold=80,
            minLineLength=30,
            maxLineGap=10
        )
        
        if lines is not None:
            # Filter and merge similar lines
            filtered_lines = self._filter_lines(lines)
            self.lines = filtered_lines
            print(f"Detected {len(self.lines)} lines")
        else:
            self.lines = []
            print("No lines detected")
            
        return self.lines
    
    def _filter_lines(self, lines, angle_threshold=5, distance_threshold=10):
        """Filter and merge similar lines"""
        if lines is None or len(lines) == 0:
            return []
            
        filtered_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle and length
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            # Only keep significant lines
            if length > 20:
                # Normalize angle to 0-180 range
                if angle < 0:
                    angle += 180
                
                filtered_lines.append({
                    'coords': (x1, y1, x2, y2),
                    'angle': angle,
                    'length': length
                })
        
        # Group lines by angle (horizontal vs vertical)
        horizontal_lines = [l for l in filtered_lines if abs(l['angle']) < angle_threshold or abs(l['angle'] - 180) < angle_threshold]
        vertical_lines = [l for l in filtered_lines if abs(l['angle'] - 90) < angle_threshold]
        
        print(f"Horizontal lines: {len(horizontal_lines)}")
        print(f"Vertical lines: {len(vertical_lines)}")
        
        return filtered_lines
    
    def detect_boxes(self):
        """Detect rectangular regions/boxes"""
        if self.processed_image is None:
            self.preprocess_image()
            
        # Find contours
        contours, _ = cv2.findContours(
            self.processed_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        boxes = []
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular (4 vertices)
            if len(approx) >= 4:
                area = cv2.contourArea(contour)
                if area > 500:  # Filter out very small regions
                    x, y, w, h = cv2.boundingRect(contour)
                    boxes.append({
                        'rect': (x, y, w, h),
                        'area': area,
                        'contour': contour
                    })
        
        # Alternative method: Use morphological operations
        kernel = np.ones((3,3), np.uint8)
        closed = cv2.morphologyEx(self.processed_image, cv2.MORPH_CLOSE, kernel)
        
        # Find contours again after closing
        contours2, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours2:
            area = cv2.contourArea(contour)
            if area > 1000:  # Larger threshold for closed shapes
                x, y, w, h = cv2.boundingRect(contour)
                # Check aspect ratio to ensure it's somewhat rectangular
                aspect_ratio = w / h if h > 0 else 0
                if 0.1 < aspect_ratio < 10:  # Reasonable aspect ratio
                    boxes.append({
                        'rect': (x, y, w, h),
                        'area': area,
                        'contour': contour
                    })
        
        # Remove duplicates based on position similarity
        unique_boxes = []
        for box in boxes:
            x, y, w, h = box['rect']
            is_duplicate = False
            for existing in unique_boxes:
                ex, ey, ew, eh = existing['rect']
                if abs(x - ex) < 20 and abs(y - ey) < 20:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_boxes.append(box)
        
        self.boxes = unique_boxes
        print(f"Detected {len(self.boxes)} rectangular regions")
        return self.boxes
    
    def analyze_composition(self):
        """Analyze the overall composition"""
        lines = self.detect_lines()
        boxes = self.detect_boxes()
        
        # Categorize lines
        horizontal_lines = [l for l in lines if abs(l['angle']) < 10 or abs(l['angle'] - 180) < 10]
        vertical_lines = [l for l in lines if abs(l['angle'] - 90) < 10]
        
        analysis = {
            'total_lines': len(lines),
            'horizontal_lines': len(horizontal_lines),
            'vertical_lines': len(vertical_lines),
            'total_boxes': len(boxes),
            'box_areas': [box['area'] for box in boxes]
        }
        
        return analysis
    
    def visualize_results(self):
        """Visualize the detection results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0, 0].imshow(self.original_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Processed edges
        axes[0, 1].imshow(self.processed_image, cmap='gray')
        axes[0, 1].set_title('Edge Detection')
        axes[0, 1].axis('off')
        
        # Lines detection
        line_image = self.original_rgb.copy()
        if self.lines:
            for line in self.lines:
                x1, y1, x2, y2 = line['coords']
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        axes[1, 0].imshow(line_image)
        axes[1, 0].set_title(f'Detected Lines ({len(self.lines)})')
        axes[1, 0].axis('off')
        
        # Boxes detection
        box_image = self.original_rgb.copy()
        if self.boxes:
            for i, box in enumerate(self.boxes):
                x, y, w, h = box['rect']
                cv2.rectangle(box_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(box_image, str(i+1), (x+5, y+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        axes[1, 1].imshow(box_image)
        axes[1, 1].set_title(f'Detected Boxes ({len(self.boxes)})')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """Generate a detailed analysis report"""
        analysis = self.analyze_composition()
        
        report = f"""
GEOMETRIC IMAGE ANALYSIS REPORT
{'='*50}
the def is een is een eigen doom van de de talk
name the def total rechtangular regions: {analysis['total_lines']}
the name of the basic design is a good 
the main idea is good to make that happen en that can be good to know that dit werkt.


LINES ANALYSIS:
- Total lines detected: {analysis['total_lines']}
- Horizontal lines: {analysis['horizontal_lines']}
- Vertical lines: {analysis['vertical_lines']}

BOXES ANALYSIS:
- Total rectangular regions: {analysis['total_boxes']}
- Average box area: {np.mean(analysis['box_areas']):.0f} pixels
- Largest box area: {max(analysis['box_areas']) if analysis['box_areas'] else 0:.0f} pixels
- Smallest box area: {min(analysis['box_areas']) if analysis['box_areas'] else 0:.0f} pixels

COMPOSITION CHARACTERISTICS:
- Grid-like structure: {'Yes' if analysis['horizontal_lines'] > 2 and analysis['vertical_lines'] > 2 else 'No'}
- Geometric style: {'Mondrian-like' if analysis['total_lines'] > 5 and analysis['total_boxes'] > 3 else 'Simple geometric'}
"""
        
        print(report)
        return report

# Usage example
def main():
    # Initialize the analyzer
    analyzer = GeometricImageAnalyzer('C:\\Users\\Sohrab\\evml-evd3\\Data\\1.jpeg')  # Replace with actual path
    
    try:
        # Load and analyze the image
        analyzer.load_image()
        
        # Run the analysis
        print("Analyzing image...")
        report = analyzer.generate_report()
        
        # Visualize results
        analyzer.visualize_results()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure to:")
        print("1. Install required packages: pip install opencv-python matplotlib numpy")
        print("2. Update the image path to point to your actual image file")

if __name__ == "__main__":
    main()