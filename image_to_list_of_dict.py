import warnings
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import uuid
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

warnings.filterwarnings("ignore", category=FutureWarning)

def generate_face_id(num_char=8):
    """Generate a unique face ID using UUID."""
    return uuid.uuid4().hex[:num_char]

def image_to_face(image, min_face_size=20, face_size=(160, 160), threshold_p_val=0.9, device='cpu'):
    """Detect faces in an image and return face tensors, bounding boxes, and probabilities."""
    mtcnn = MTCNN(keep_all=True, min_face_size=min_face_size, device=device)
    transform = transforms.ToTensor()

    # Convert image to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Detect faces and probabilities in the image
    bounds, p_vals = mtcnn.detect(image)

    valid_bounds = []
    valid_p_vals = []

    # Filter out low-confidence detections
    if bounds is not None:
        for box, p_val in zip(bounds, p_vals):
            if p_val > threshold_p_val:
                valid_bounds.append(box)
                valid_p_vals.append(p_val)

    face_tensors = []
    boundary_boxs = []

    # Process each valid bounding box
    if valid_bounds is not None:
        for box in valid_bounds:
            box = [
                max(0, box[0] - 20),  
                max(0, box[1] - 30),  
                min(image.width, box[2] + 20),  
                min(image.height, box[3] + 20)  
            ]
            face = image.crop(tuple(box)).resize(face_size)  # Crop and resize face
            face_tensor = transform(face).to(device)

            face_tensors.append(face_tensor)
            boundary_boxs.append(box)

    return face_tensors, boundary_boxs, valid_p_vals

def face_tensor_to_embedding(face_tensors, resnet_model='vggface2', device='cpu'):
    """Convert face tensors to embeddings using a pretrained model."""
    resnet = InceptionResnetV1(pretrained=resnet_model, device=device).eval()
    face_embeddings = []

    # Disable gradients for inference
    with torch.inference_mode():
        for face_tensor in face_tensors:
            if face_tensor.shape[1] < 10 or face_tensor.shape[2] < 10:
                continue  # Skip too small tensors

            face_embedding = resnet(face_tensor.unsqueeze(0))  # Compute face embedding
            face_embeddings.append(face_embedding.cpu())

    return face_embeddings

def image_to_face_list(image_path, device='cpu'):
    """Detect faces in an image and return a list of dictionaries with face details."""
    image = Image.open(image_path)  # Open the image using PIL

    face_list = []  # List to store dictionaries with face details

    # Detect faces and get face tensors, bounding boxes, and probabilities
    face_tensors, bounds, p_vals = image_to_face(image, device=device, min_face_size=20, threshold_p_val=0.95)

    # Convert face tensors to face embeddings
    face_embeddings = face_tensor_to_embedding(face_tensors, resnet_model='vggface2', device=device)

    # For each face embedding, generate a unique face ID and store its details in a dictionary
    for i, face_embedding in enumerate(face_embeddings):
        face_id = generate_face_id()  # Generate a unique ID for each face

        face_dict = {
            'id': face_id,  # Unique face ID
            'embedding': face_embedding.tolist(),  # Convert tensor to list for JSON serialization
            'bounding_box': bounds[i],  # Convert bounding box to list
            'probability': p_vals[i]  # Detection probability of the face
        }

        face_list.append(face_dict)

    return face_list  # Return the list of detected faces and embeddings

def load_images_from_directory(directory, batch_size=1, image_size=(160, 160), shuffle=True):
    """Load images from a directory into a DataLoader."""
    transform = transforms.Compose([
        transforms.Resize(image_size),  # Resize images to a fixed size
        transforms.ToTensor(),  # Convert images to tensors
    ])

    valid_images = []  # List to hold valid image files

    # Collect valid image paths
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')):
            valid_images.append(filepath)

    if not valid_images:
        raise ValueError("No valid image files found in the directory.")

    # Create a custom dataset
    dataset = [(transform(Image.open(img)), img) for img in valid_images]  # Load and transform images

    # Use a DataLoader to load the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

def get_all_face_list(directory):
    """Get a list of all detected faces from images in the specified directory."""
    dataloader = load_images_from_directory(directory, batch_size=1)

    all_faces = []  # List to hold all faces from all images

    # Process each image in the DataLoader
    for images, file_paths in dataloader:
        for file_path in file_paths:  # Iterate over file paths
            face_list = image_to_face_list(file_path, device='cuda' if torch.cuda.is_available() else 'cpu')  # Pass the file path
            all_faces.extend(face_list)  # Add the detected faces to the all_faces list

    return all_faces

if __name__ == "__main__":
    directory = "/home/sajid/Work/ResilientSage/FaceNet/Images"
    face_list = get_all_face_list(directory)
    print(face_list[0])  # Print the first face's details
