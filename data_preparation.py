import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import matplotlib.pyplot as plt

class BreakHisLoader:
    def __init__(self, root_path=r"C:\Users\DELL\Downloads\BreaKHis_v1\BreaKHis_v1"):
        self.root_path = os.path.normpath(root_path)
        self.classes = {'benign': 0, 'malignant': 1}
        self.magnifications = ['40X', '100X', '200X', '400X']
        self.metadata = []
        
        # Verify dataset structure
        self._verify_dataset_structure()

    def _verify_dataset_structure(self):
        """Verify the dataset directory structure exists"""
        required_path = os.path.join(self.root_path, "histology_slides", "breast")
        if not os.path.exists(required_path):
            raise FileNotFoundError(
                f"Dataset directory not found at: {required_path}\n"
                "Please ensure:\n"
                "1. The BreaKHis dataset is downloaded\n"
                "2. It's extracted to the correct location\n"
                "3. The directory structure matches the expected format"
            )
        
        # Check for at least some image files
        found_images = False
        for root, _, files in os.walk(required_path):
            if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
                found_images = True
                break
                
        if not found_images:
            raise FileNotFoundError(
                f"No image files found in {required_path}\n"
                "Expected to find .png/.jpg/.jpeg files in subdirectories"
            )

    def parse_path(self, path):
        """Parse metadata from image path with better error handling"""
        path = os.path.normpath(path)
        parts = path.split(os.sep)

        try:
            # Extract the class (benign or malignant)
            if 'benign' in parts:
                class_name = 'benign'
            elif 'malignant' in parts:
                class_name = 'malignant'
            else:
                raise ValueError(f"Path doesn't contain 'benign' or 'malignant': {path}")

            # Extract the magnification (e.g., '40X')
            magnification = None
            for part in parts:
                if 'X' in part:
                    magnification = part
                    break

            if not magnification:
                raise ValueError(f"Magnification level not found in path: {path}")

            subtype = parts[-3]  # Three levels up from the file name
            slide_id = parts[-2]  # Two levels up from the file name

            return {
                'class': class_name,
                'subtype': subtype,
                'slide_id': slide_id,
                'magnification': magnification,
                'filename': parts[-1],
                'full_path': path
            }
        except IndexError as e:
            raise ValueError(f"Error parsing path {path}: {str(e)}")
        except ValueError as e:
            raise ValueError(f"Value error in path {path}: {str(e)}")

    def load_image(self, args):
        """Load and preprocess a single image"""
        path, target_size = args
        try:
            img = cv2.imread(path)
            if img is None:
                raise ValueError(f"Could not read image at {path}")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            return img, path
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            return None, None

    def load_dataset(self, target_size=(128, 128), max_workers=4, test_size=0.2, max_samples=None):
        """Load dataset with better error handling and progress tracking"""
        print(f"Loading dataset from: {self.root_path}")
        search_path = os.path.join(self.root_path, "histology_slides", "breast")

        # Find all image paths
        image_paths = []
        for root, _, files in os.walk(search_path):
            if any(mag in root for mag in self.magnifications):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')): 
                        full_path = os.path.join(root, file)
                        image_paths.append(full_path)
        
        if not image_paths:
            raise ValueError(
                f"No images found in {search_path}\n"
                "Check that:\n"
                "1. The dataset is properly extracted\n"
                "2. Images exist in magnification folders (40X, 100X, etc.)\n"
                "3. You have read permissions"
            )

        # Optionally limit samples for testing
        if max_samples:
            image_paths = image_paths[:max_samples]

        print(f"\nFound {len(image_paths)} images. Loading...")

        # Load images in parallel
        images = []
        valid_paths = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(self.load_image, [(p, target_size) for p in image_paths]),
                total=len(image_paths),
                desc="Loading images"
            ))

        # Process results
        for img, path in results:
            if img is not None:
                images.append(img)
                valid_paths.append(path)
                try:
                    self.metadata.append(self.parse_path(path))
                except ValueError as e:
                    print(f"Skipping metadata for {path}: {str(e)}")

        if not images:
            raise ValueError("No images were successfully loaded")

        # Convert to numpy array and normalize
        images = np.array(images, dtype=np.float32) / 255.0

        # Create labels
        labels = np.array([self.classes[m['class']] for m in self.metadata])

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, 
            test_size=test_size, 
            random_state=42, 
            stratify=labels
        )

        # Create metadata dataframe
        df_meta = pd.DataFrame(self.metadata)
        df_meta['label'] = labels

        return X_train, X_test, y_train, y_test, df_meta

def visualize_samples(images, labels, metadata, num_samples=5):
    """Visualize sample images with metadata"""
    plt.figure(figsize=(15, 5))
    for i in range(min(num_samples, len(images))):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(images[i])
        title = (f"{metadata.iloc[i]['class']}\n"
                 f"{metadata.iloc[i]['subtype']}\n"
                 f"{metadata.iloc[i]['magnification']}") 
        plt.title(title, fontsize=8)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    try:
        print("Current Working Directory:", os.getcwd())

        # Initialize loader with fixed dataset path
        loader = BreakHisLoader(root_path=r"C:\Users\DELL\Downloads\BreaKHis_v1\BreaKHis_v1")
        
        # Load dataset
        X_train, X_test, y_train, y_test, df_meta = loader.load_dataset(
            target_size=(128, 128),
            max_workers=4,
            test_size=0.2
        )
        
        print("\n=== Dataset Loaded Successfully ===")
        print(f"Total images: {len(df_meta)}")
        print(f"Training set: {len(X_train)} images ({len(X_train)/len(df_meta):.1%})")
        print(f"Test set: {len(X_test)} images ({len(X_test)/len(df_meta):.1%})")

        print("\n=== Class Distribution ===")
        class_dist = pd.DataFrame({
            'Train': pd.Series(y_train).value_counts(),
            'Test': pd.Series(y_test).value_counts()
        })
        class_dist.index = ['benign (0)', 'malignant (1)']
        print(class_dist)

        print("\n=== Magnification Distribution ===")
        print(df_meta['magnification'].value_counts())

        print("\n=== Subtype Distribution ===")
        print(df_meta['subtype'].value_counts())

        print("\nVisualizing training samples...")
        train_indices = np.random.choice(len(X_train), 5, replace=False)
        visualize_samples(X_train[train_indices], y_train[train_indices],
                         df_meta.iloc[train_indices])
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Verify the dataset is at C:/Users/DELL/Downloads/BreaKHis_v1/BreaKHis_v1")
        print("2. Ensure the dataset is extracted properly")
        print("3. Check directory structure contains 'benign' and 'malignant' subfolders")
        print("4. Confirm images are in .png/.jpg/.jpeg format")
        print("5. Check read permissions for the files")