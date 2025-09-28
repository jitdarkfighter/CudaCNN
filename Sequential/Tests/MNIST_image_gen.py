import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import random
import math
import os

class MNISTStyleGenerator:
    def __init__(self, image_size=28):
        self.image_size = image_size
        
    def add_noise(self, image, noise_level=0.1):
        """Add random noise to make the image look more handwritten"""
        noise = np.random.normal(0, noise_level, image.shape)
        noisy_image = image + noise
        return np.clip(noisy_image, 0, 1)
    
    def apply_rotation(self, image, max_angle=15):
        """Apply random rotation to simulate handwriting variation"""
        angle = random.uniform(-max_angle, max_angle)
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        rotated = pil_image.rotate(angle, fillcolor=0, expand=False)
        return np.array(rotated) / 255.0
    
    def apply_translation(self, image, max_shift=2):
        """Apply random translation"""
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        
        translated = np.zeros_like(image)
        h, w = image.shape
        
        # Calculate bounds to avoid index errors
        src_y_start = max(0, -shift_y)
        src_y_end = min(h, h - shift_y)
        src_x_start = max(0, -shift_x)
        src_x_end = min(w, w - shift_x)
        
        dst_y_start = max(0, shift_y)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = max(0, shift_x)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)
        
        translated[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            image[src_y_start:src_y_end, src_x_start:src_x_end]
        
        return translated
    
    def generate_digit_0(self):
        """Generate digit 0"""
        image = np.zeros((self.image_size, self.image_size))
        center = self.image_size // 2
        
        # Create more realistic oval shape similar to MNIST
        for y in range(self.image_size):
            for x in range(self.image_size):
                # Ellipse equation - make it more MNIST-like
                dx = (x - center) / 8.0  # Width control
                dy = (y - center) / 10.0  # Height control
                dist = dx*dx + dy*dy
                
                # Create thick border with smoother edges
                if 0.3 < dist < 0.95:
                    # Smooth falloff for anti-aliasing effect
                    if dist < 0.6:
                        intensity = 1.0
                    else:
                        intensity = 1.0 - (dist - 0.6) / 0.35
                    
                    image[y, x] = max(0, min(1.0, intensity))
                elif dist <= 0.3:
                    # Inner part should be empty (black)
                    image[y, x] = 0.0
        
        return image
    
    def generate_digit_1(self):
        """Generate digit 1"""
        image = np.zeros((self.image_size, self.image_size))
        center_x = self.image_size // 2 + random.randint(-1, 1)
        
        # Main vertical stroke
        thickness = 3
        start_y = 4
        end_y = self.image_size - 4
        
        for y in range(start_y, end_y):
            for dx in range(-thickness//2, thickness//2 + 1):
                x = center_x + dx
                if 0 <= x < self.image_size:
                    # Smooth edges
                    intensity = 1.0 - abs(dx) * 0.2
                    image[y, x] = max(image[y, x], intensity)
        
        # Top diagonal stroke (more MNIST-like)
        for i in range(6):
            x = center_x - 4 + i
            y = start_y + i // 2
            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                image[y, x] = max(image[y, x], 0.9)
        
        # Small base
        for x in range(center_x - 2, center_x + 3):
            if 0 <= x < self.image_size:
                image[end_y - 1, x] = max(image[end_y - 1, x], 0.8)
        
        return image
    
    def generate_digit_2(self):
        """Generate digit 2"""
        image = np.zeros((self.image_size, self.image_size))
        
        # Top arc (more controlled)
        center_x = self.image_size // 2
        for angle in np.linspace(-0.3, np.pi + 0.3, 40):
            x = int(center_x + 7 * np.cos(angle))
            y = int(8 + 4 * np.sin(angle))
            self._draw_thick_point(image, x, y, 2)
        
        # Right vertical part
        for y in range(12, 18):
            self._draw_thick_point(image, center_x + 7, y, 1)
        
        # Diagonal sweep
        for i in range(12):
            t = i / 11.0
            x = int(center_x + 7 - t * 12)
            y = int(18 + t * 6)
            self._draw_thick_point(image, x, y, 2)
        
        # Bottom horizontal line
        for x in range(center_x - 6, center_x + 8):
            self._draw_thick_point(image, x, 24, 2)
        
        return image
    
    def generate_digit_3(self):
        """Generate digit 3"""
        image = np.zeros((self.image_size, self.image_size))
        center_x = self.image_size // 2
        
        # Top arc
        for angle in np.linspace(-np.pi/2, np.pi/2, 20):
            x = int(center_x + 6 * np.cos(angle))
            y = int(8 + 4 * np.sin(angle))
            self._draw_thick_point(image, x, y, 2)
        
        # Middle horizontal line
        for x in range(center_x, center_x + 8):
            self._draw_thick_point(image, x, self.image_size // 2, 1)
        
        # Bottom arc
        for angle in np.linspace(-np.pi/2, np.pi/2, 20):
            x = int(center_x + 6 * np.cos(angle))
            y = int(self.image_size - 8 + 4 * np.sin(angle))
            self._draw_thick_point(image, x, y, 2)
        
        return image
    
    def generate_digit_4(self):
        """Generate digit 4"""
        image = np.zeros((self.image_size, self.image_size))
        
        # Vertical line (left)
        x = 8
        for y in range(5, 18):
            self._draw_thick_point(image, x, y, 2)
        
        # Vertical line (right)
        x = 18
        for y in range(3, self.image_size - 3):
            self._draw_thick_point(image, x, y, 2)
        
        # Horizontal line
        y = 17
        for x in range(6, 22):
            self._draw_thick_point(image, x, y, 2)
        
        return image
    
    def generate_digit_5(self):
        """Generate digit 5"""
        image = np.zeros((self.image_size, self.image_size))
        
        # Top horizontal line
        for x in range(7, 20):
            self._draw_thick_point(image, x, 5, 2)
        
        # Left vertical line
        for y in range(5, 15):
            self._draw_thick_point(image, 7, y, 2)
        
        # Middle horizontal line
        for x in range(7, 18):
            self._draw_thick_point(image, x, 14, 1)
        
        # Bottom arc
        center_x = 15
        center_y = 18
        for angle in np.linspace(0, np.pi, 15):
            x = int(center_x + 6 * np.cos(angle))
            y = int(center_y + 4 * np.sin(angle))
            self._draw_thick_point(image, x, y, 2)
        
        return image
    
    def generate_digit_6(self):
        """Generate digit 6"""
        image = np.zeros((self.image_size, self.image_size))
        center = self.image_size // 2
        
        # Top curved part coming from right
        for angle in np.linspace(-np.pi/2, np.pi/2, 25):
            x = int(center - 2 + 5 * np.cos(angle))
            y = int(8 + 6 * np.sin(angle))
            self._draw_thick_point(image, x, y, 2)
        
        # Left vertical part
        for y in range(8, 22):
            self._draw_thick_point(image, center - 6, y, 2)
        
        # Bottom loop (closed circle)
        for angle in np.linspace(0, 2*np.pi, 40):
            x = int(center - 1 + 4.5 * np.cos(angle))
            y = int(18 + 3.5 * np.sin(angle))
            self._draw_thick_point(image, x, y, 2)
        
        return image
    
    def generate_digit_7(self):
        """Generate digit 7"""
        image = np.zeros((self.image_size, self.image_size))
        
        # Top horizontal line (more typical MNIST 7 style)
        for x in range(8, 20):
            self._draw_thick_point(image, x, 5, 2)
        
        # Main diagonal stroke (more pronounced)
        for i in range(20):
            t = i / 19.0
            x = int(19 - t * 6)
            y = int(6 + t * 17)
            self._draw_thick_point(image, x, y, 2)
        
        # Small horizontal crossbar (common in MNIST 7s)
        for x in range(10, 16):
            self._draw_thick_point(image, x, 12, 1)
        
        return image
    
    def generate_digit_8(self):
        """Generate digit 8"""
        image = np.zeros((self.image_size, self.image_size))
        center = self.image_size // 2
        
        # Top circle
        for angle in np.linspace(0, 2*np.pi, 30):
            x = int(center + 5 * np.cos(angle))
            y = int(9 + 3 * np.sin(angle))
            self._draw_thick_point(image, x, y, 1)
        
        # Bottom circle
        for angle in np.linspace(0, 2*np.pi, 35):
            x = int(center + 6 * np.cos(angle))
            y = int(19 + 4 * np.sin(angle))
            self._draw_thick_point(image, x, y, 2)
        
        return image
    
    def generate_digit_9(self):
        """Generate digit 9"""
        image = np.zeros((self.image_size, self.image_size))
        center = self.image_size // 2
        
        # Top circle (the loop of 9)
        for angle in np.linspace(0, 2*np.pi, 35):
            x = int(center + 4 * np.cos(angle))
            y = int(10 + 3.5 * np.sin(angle))
            self._draw_thick_point(image, x, y, 2)
        
        # Right vertical stroke going down
        for y in range(14, 22):
            self._draw_thick_point(image, center + 4, y, 2)
        
        # Bottom curl
        for angle in np.linspace(-np.pi/2, 0, 10):
            x = int(center + 4 + 3 * np.cos(angle))
            y = int(22 + 3 * np.sin(angle))
            self._draw_thick_point(image, x, y, 2)
        
        return image
    
    def _draw_thick_point(self, image, x, y, thickness):
        """Helper function to draw a thick point with smooth anti-aliasing"""
        for dy in range(-thickness-1, thickness + 2):
            for dx in range(-thickness-1, thickness + 2):
                px, py = x + dx, y + dy
                if 0 <= px < self.image_size and 0 <= py < self.image_size:
                    distance = math.sqrt(dx*dx + dy*dy)
                    if distance <= thickness + 0.5:
                        # Smooth falloff for anti-aliasing
                        if distance <= thickness - 0.5:
                            intensity = 1.0
                        else:
                            intensity = max(0, 1.0 - (distance - thickness + 0.5))
                        image[py, px] = max(image[py, px], intensity)
    
    def generate_digit(self, digit, add_variations=True):
        """Generate a single digit image"""
        if digit == 0:
            image = self.generate_digit_0()
        elif digit == 1:
            image = self.generate_digit_1()
        elif digit == 2:
            image = self.generate_digit_2()
        elif digit == 3:
            image = self.generate_digit_3()
        elif digit == 4:
            image = self.generate_digit_4()
        elif digit == 5:
            image = self.generate_digit_5()
        elif digit == 6:
            image = self.generate_digit_6()
        elif digit == 7:
            image = self.generate_digit_7()
        elif digit == 8:
            image = self.generate_digit_8()
        elif digit == 9:
            image = self.generate_digit_9()
        else:
            raise ValueError("Digit must be between 0 and 9")
        
        if add_variations:
            # Apply subtle transformations to make it look more realistic
            if random.random() < 0.3:  # 30% chance of small rotation
                image = self.apply_rotation(image, 5)
            if random.random() < 0.2:  # 20% chance of small translation
                image = self.apply_translation(image, 1)
            if random.random() < 0.4:  # 40% chance of light noise
                image = self.add_noise(image, noise_level=random.uniform(0.02, 0.08))
        
        # Ensure values are in [0, 1] range
        image = np.clip(image, 0, 1)
        return image
    
    def generate_batch(self, digit, num_samples=10):
        """Generate multiple samples of the same digit"""
        samples = []
        for _ in range(num_samples):
            sample = self.generate_digit(digit)
            samples.append(sample)
        return np.array(samples)
    
    def save_image(self, image, filename):
        """Save image to file"""
        os.makedirs("Test-images", exist_ok=True)
        # Convert to 0-255 range and save PNG
        img_array = (image * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img.save(f"Test-images/{filename}")
        
        # Also save as raw binary file for C++ testing
        base_name = os.path.splitext(filename)[0]
        raw_filename = f"Test-images/{base_name}.raw"
        # Save as normalized float values (0-1 range)
        image.astype(np.float32).tofile(raw_filename)
    
    def display_samples(self, digit, num_samples=5):
        """Display multiple samples of a digit"""
        samples = self.generate_batch(digit, num_samples)
        
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
        if num_samples == 1:
            axes = [axes]
            
        for i, ax in enumerate(axes):
            ax.imshow(samples[i], cmap='gray')
            ax.set_title(f'Digit {digit} - Sample {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    generator = MNISTStyleGenerator()
    
    # Generate and display samples for each digit
    print("Generating MNIST-style digit samples...")
    
    # for digit in range(10):
    #     print(f"Generating samples for digit {digit}")
    #     generator.display_samples(digit, num_samples=5)
    
    for digit in range(10):
        # Generate without variations for more consistent testing
        single_digit = generator.generate_digit(digit, add_variations=True)
        generator.save_image(single_digit, f"mnist_style_{digit}.png")
    # # Generate a single digit and save it
    # single_digit = generator.generate_digit(5)
    # generator.save_image(single_digit, "mnist_style_5.png")
    # print("Saved single digit 5 as 'mnist_style_5.png'")
    
    # # Generate a batch of digits
    # batch = generator.generate_batch(3, num_samples=20)
    # print(f"Generated batch of 20 digit-3 samples with shape: {batch.shape}")