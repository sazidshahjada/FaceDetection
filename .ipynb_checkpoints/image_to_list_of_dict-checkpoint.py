import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore FutureWarnings to clean up output

import uuid  # Used to generate unique face IDs
import torch  # PyTorch for tensor operations
import numpy as np  # NumPy for handling arrays (not used in this snippet, but commonly imported)
from PIL import Image  # Python Imaging Library to handle image loading and manipulation
from torchvision import transforms  # Torchvision for transformations (e.g., converting PIL images to tensors)
from facenet_pytorch import MTCNN, InceptionResnetV1  # Models for face detection and face embeddings

# Function to generate a unique ID for each detected face
def generate_face_id(num_char=8):
    """
    Generates a unique face ID using UUID.
    
    Parameters:
    num_char (int): The length of the unique ID to generate (default 8 characters).
    
    Returns:
    str: A unique ID string.
    """
    return uuid.uuid4().hex[:num_char]  # Create a random UUID and take the first 'num_char' characters


# Function to detect faces in an image and convert them to face tensors
def image_to_face(image, min_face_size=20, threshold_p_val=0.9, device='cpu'):
    """
    Detects faces in an image using MTCNN and returns face tensors for further processing.
    
    Parameters:
    image (PIL.Image): The input image.
    min_face_size (int): Minimum size of faces to detect (default 20 pixels).
    threshold_p_val (float): The probability threshold for valid face detection (default 0.9).
    device (str): The device to run computations on (e.g., 'cpu' or 'cuda').
    
    Returns:
    list: A list of face tensors.
    list: A list of bounding box coordinates for each face.
    list: A list of detection probabilities for each face.
    """
    mtcnn = MTCNN(keep_all=True, min_face_size=min_face_size, device=device)  # Initialize MTCNN for face detection
    transform = transforms.ToTensor()  # Convert PIL images to torch tensors

    # Convert the image to RGB if it's not (useful for images with transparency or different color modes)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Detect faces and their probabilities in the image
    bounds, p_vals = mtcnn.detect(image)

    valid_bounds = []  # List to store valid bounding boxes (faces with probability > threshold)
    valid_p_vals = []  # List to store corresponding probabilities

    # Check detected faces' probabilities and filter out low-confidence faces
    if bounds is not None:
        for box, p_val in zip(bounds, p_vals):
            if p_val > threshold_p_val:  # Only consider faces with probability above threshold
                valid_bounds.append(box)
                valid_p_vals.append(p_val)

    face_tensors = []  # List to store face tensors
    boundary_boxs = []  # List to store the adjusted bounding boxes

    # Process each valid bounding box and create face tensors
    if valid_bounds is not None:
        for i, box in enumerate(valid_bounds):
            # Adjust bounding box to ensure it stays within image bounds and adds some margin
            box = [
                max(0, box[0] - 20),  # Adjust left boundary
                max(0, box[1] - 30),  # Adjust top boundary
                min(image.width, box[2] + 20),  # Adjust right boundary
                min(image.height, box[3] + 20)  # Adjust bottom boundary
            ]
            
            # Crop the image using the bounding box
            face = image.crop(tuple(box))
            
            # Resize the face to a fixed size (160x160) for input to InceptionResnetV1 model
            face = face.resize((160, 160))
            
            # Convert the cropped face image to a tensor
            face_tensor = transform(face).to(device)
            
            # Append the face tensor and bounding box to the respective lists
            face_tensors.append(face_tensor)
            boundary_boxs.append(box)

    # Return the list of face tensors, bounding boxes, and detection probabilities
    return face_tensors, boundary_boxs, valid_p_vals


# Function to convert face tensors to face embeddings
def face_tensor_to_embedding(face_tensors, resnet_model='vggface2', device='cpu'):
    """
    Converts face tensors to face embeddings using the InceptionResnetV1 model.
    
    Parameters:
    face_tensors (list): A list of face tensors.
    resnet_model (str): The pretrained model to use for face embeddings (default is 'vggface2').
                        Expected values: 'vggface2' or 'casia-webface'
    device (str): The device to run computations on (e.g., 'cpu' or 'cuda').
    
    Returns:
    list: A list of face embeddings (tensor).
    """
    # Load the pretrained InceptionResnetV1 model for face embeddings
    resnet = InceptionResnetV1(pretrained=resnet_model, device=device).eval()

    face_embeddings = []  # List to store the face embeddings

    # Disable gradients for inference (faster)
    with torch.inference_mode():
        for face_tensor in face_tensors:
            # Ensure that the face tensor is not too small (check the height and width)
            if face_tensor.shape[1] < 10 or face_tensor.shape[2] < 10:
                continue  # Skip the face if it's too small

            # Compute the face embedding by passing the face tensor through the model
            face_embedding = resnet(face_tensor.unsqueeze(0))
            
            # Append the face embedding to the list
            face_embeddings.append(face_embedding.cpu())
    
    return face_embeddings


# Function to read an image and generate a list of detected faces and their embeddings
def image_to_face_list(image_path, device='cpu'):
    """
    Reads an image file, detects faces, and generates face embeddings for each detected face.
    
    Parameters:
    image_path (str): Path to the input image file.
    device (str): The device to run computations on (e.g., 'cpu' or 'cuda').
    
    Returns:
    list: A list of dictionaries where each dictionary contains the face ID, embedding, bounding box, and probability.
    """
    # Open the image using PIL
    image = Image.open(image_path)

    face_list = []  # List to store dictionaries with face details

    # Detect faces and get face tensors, bounding boxes, and probabilities
    face_tensors, bounds, p_vals = image_to_face(image, device=device, min_face_size=20, threshold_p_val=0.95)

    # Convert face tensors to face embeddings
    face_embeddings = face_tensor_to_embedding(face_tensors,resnet_model='vggface2', device=device)

    # For each face embedding, generate a unique face ID and store its details in a dictionary
    for i, face_embedding in enumerate(face_embeddings):
        face_id = generate_face_id()  # Generate a unique ID for each face

        # Create a dictionary with the face's details
        face_dict = {
            'id': face_id,  # Unique face ID
            'embedding': face_embedding,  # Face embedding (tensor)
            'boundary_box': bounds[i],  # Bounding box of the face in the image
            'probability': p_vals[i]  # Detection probability of the face
        }

        # Append the face dictionary to the list
        face_list.append(face_dict)

    return face_list  # Return the list of detected faces and embeddings


# Main function to run the face detection and embedding extraction process
if __name__ == "__main__":
    # Path to the input image
    image_path = "/home/sajid/Work/ResilientSage/FaceNet/Images/grp.jpg"
    
    # Call the image_to_face_list function to get face details (use 'cuda' if available)
    face_list = image_to_face_list(image_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Print the list of detected faces and their details
    print(face_list)
