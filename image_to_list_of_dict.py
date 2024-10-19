import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Suppress future warnings

import os
import json
import uuid
import torch
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1

# Function to generate a unique ID for each detected face
def generate_face_id(num_char=8):
    return uuid.uuid4().hex[:num_char]  # Generate a random 8-character hexadecimal ID

# Function to check if a file is a valid image by attempting to open it
def is_image_file(filepath):
    try:
        Image.open(filepath)  # Try to open the file as an image
        return True
    except:
        return False  # Return False if the file cannot be opened as an image

# Custom dataset class for loading images from a directory
class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        # List of valid image files in the directory
        self.image_files = [f for f in os.listdir(image_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and is_image_file(os.path.join(image_dir, f))]
        self.transform = transform
        # Raise an error if no valid image files are found
        if len(self.image_files) == 0:
            raise ValueError(f"No valid image files found in directory: {image_dir}")

    def __len__(self):
        return len(self.image_files)  # Return the number of image files

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])  # Get the image file path
        image = Image.open(image_path).convert("RGB")  # Open the image and convert it to RGB
        
        if self.transform:
            image = self.transform(image)  # Apply transformations if any
        
        return image, self.image_files[idx]  # Return the transformed image and its filename

# Function to create a DataLoader for unlabelled images
def create_unlabel_dataloader(image_dir, image_size=(500, 500), batch_size=4, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(image_size),  # Resize images
        transforms.ToTensor(),  # Convert images to tensors
    ])
    dataset = ImageDataset(image_dir=image_dir, transform=transform)  # Create the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)  # Create DataLoader
    return dataloader

# Function to detect faces in a given image
def detect_face(image: torch.Tensor, image_size=(500, 500), min_face_size=20, threshold_p_val=0.9, device='cpu'):
    mtcnn = MTCNN(keep_all=True, image_size=image_size, min_face_size=min_face_size, device=device)  # Initialize MTCNN
    transform = transforms.ToTensor()  # Define a transformation to convert images to tensors

    # Convert the tensor image to a PIL image for face detection
    image_np = image.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
    image_np = (image_np * 255).byte().numpy()  # Convert to a numpy array with pixel values in range [0, 255]
    pillow_image = Image.fromarray(image_np)  # Create a PIL image from the numpy array

    # Detect faces and probability values
    face_boundaries, probabilities = mtcnn.detect(pillow_image)

    # If no faces are detected, return empty lists
    if face_boundaries is None or len(face_boundaries) == 0:
        return [], [], []

    valid_face_boundaries = []  # Store valid bounding boxes
    face_probabilities = []  # Store corresponding probabilities

    # Filter faces based on probability threshold
    for box, p_val in zip(face_boundaries, probabilities):
        if p_val > threshold_p_val:
            valid_face_boundaries.append(box)
            face_probabilities.append(p_val)

    face_tensors = []  # List to hold cropped face tensors
    boundary_boxes = []  # List to hold bounding boxes for faces

    # Loop over detected valid faces
    for i, box in enumerate(valid_face_boundaries):
        # Add padding to face bounding box
        (box[0], box[1], box[2], box[3]) = (box[0] - 20, box[1] - 30, box[2] + 20, box[3] + 20)
        
        # Ensure box coordinates stay within image dimensions
        box = [max(0, b) for b in box]
        box[2] = min(box[2], image.shape[2])
        box[3] = min(box[3], image.shape[1])

        # Crop the face from the image
        face = pillow_image.crop(tuple(box))
        
        # Resize the face to match the input size of InceptionResnetV1
        face = face.resize((160, 160))

        face_tensor = transform(face)  # Convert the face to a tensor
        
        face_tensors.append(face_tensor.unsqueeze(0))  # Add a batch dimension to the tensor
        boundary_boxes.append(box)  # Save the bounding box for the face

    return face_tensors, boundary_boxes, face_probabilities  # Return faces, bounding boxes, and probabilities

# Function to create a face embedding using InceptionResnetV1
def create_face_embedding(face_tensor: torch.Tensor, resnet_model='vggface2', device='cpu'):
    resnet = InceptionResnetV1(pretrained=resnet_model, device=device).eval()  # Load pre-trained InceptionResnetV1

    face_tensor = face_tensor.to(device)  # Move face tensor to the correct device

    # Get the face embedding without calculating gradients
    with torch.inference_mode():
        face_embedding = resnet(face_tensor)

    return face_embedding.cpu()  # Return the embedding tensor moved to CPU

# Function to process images and create JSON output for detected faces
def create_json(image_dir, image_size=(500, 500),batch_size=4, num_workers=4, device='cpu'):
    # Create a DataLoader for the images
    dataloader = create_unlabel_dataloader(
        image_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers
    )

    face_dict_list = []  # List to store face information

    # Iterate through the batches of images
    for images, _ in tqdm(dataloader, desc="Analyzing Batches"):
        for image in images:  # Iterate through images in the batch
            # Detect faces in the image
            face_tensors, face_boundaries, probabilities = detect_face(
                image, image_size=image_size,
                min_face_size=20,
                threshold_p_val=0.95,
                device=device
            )

            # Process each detected face
            for i, face_tensor in enumerate(face_tensors):
                face_dict = dict()  # Dictionary to hold face data

                # Generate face embedding
                face_embedding = create_face_embedding(
                    face_tensor.to(device),
                    resnet_model='vggface2',
                    device=device
                )
                
                face_id = generate_face_id()  # Generate unique ID for the face
                face_bound = face_boundaries[i]  # Get bounding box for the face
                p_val = probabilities[i]  # Get probability for the face detection

                # Add face information to dictionary
                face_dict['face_id'] = face_id
                face_dict['face_embedding'] = face_embedding.tolist()  # Convert embedding to list for JSON
                face_dict['face_boundary_box'] = face_bound
                face_dict['face_probability'] = p_val

                face_dict_list.append(face_dict)  # Append face data to the list


    return face_dict_list  # Return the list of face dictionaries

# Main function to run the face detection and embedding process
if __name__ == "__main__":
    import time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, otherwise use CPU

    image_dir = "/home/sajid/Work/ResilientSage/FaceNet/unsplash_face"  # Set the directory of images to process

    start = time.time()  # Record the start time

    # Run the face detection and embedding process, outputting data as a JSON-like structure
    json_data = create_json(
        image_dir=image_dir,
        image_size=(224, 224),
        device=device,
        batch_size=4,
        num_workers=4
    )

    end = time.time()  # Record the end time

    print(f"Number of faces detected: {len(json_data)}")
    print(f"Using device : {device}")
    print(f"Execution time : {end - start} seconds")  # Print the execution time

    print(json_data)
