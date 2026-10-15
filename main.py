import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

# 1. Setup Device (Use GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load Pre-trained ResNet-50 Model
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.to(device)
model.eval()  # Set model to evaluation mode

# Get the category names (labels) from the weights
categories = weights.meta["categories"]

# 3. Define Image Transformations (ImageNet standards)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 4. Initialize Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' in the video window to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # OpenCV uses BGR, PyTorch/Pillow expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    # Apply transformations and add a batch dimension [1, 3, 224, 224]
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # 5. Make Prediction (Disable gradients for faster real-time speed)
    with torch.no_grad():
        output = model(input_tensor)
    
    # Apply Softmax to raw output scores to get probability distributions [0.0 to 1.0]
    # We flatten output to 1D array using [0] to extract top predictions properly
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get the top 5 highest predictions and their index positions
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    # Clear the Fedora terminal screen text so it prints cleanly in place
    print("\033[H\033[J", end="") 
    print("=== LIVE MODEL GUESSES ===")
    
    for i in range(top5_prob.size(0)):
        label = categories[top5_catid[i].item()]
        accuracy = top5_prob[i].item() * 100
        print(f"{i+1}: {label:<25} {accuracy:.2f}%")

    # 6. Overlay the #1 Highest Guess onto the OpenCV Webcam Window
    top1_label = categories[top5_catid[0].item()]
    top1_accuracy = top5_prob[0].item() * 100
    text = f"{top1_label} ({top1_accuracy:.1f}%)"
    
    # Draw a clean dark background banner at the top of the video box
    cv2.rectangle(frame, (10, 10), (450, 50), (0, 0, 0), -1)
    
    # Overlay the text
    cv2.putText(frame, text, (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the live window frame
    cv2.imshow("PyTorch + OpenCV Object Classifier", frame)

    # Break loop cleanly if user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up window objects and webcam hardware locks safely
cap.release()
cv2.destroyAllWindows()
print("\nApplication closed successfully.")
