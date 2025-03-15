import torch
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from diffusers import StableDiffusionInpaintPipeline
import matplotlib.pyplot as plt

# Load Mask R-CNN Model
mask_rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
mask_rcnn.eval()

# Load Stable Diffusion Inpainting Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to(device)

# Function to Detect and Mask Objects
def generate_mask(image_path, threshold=0.5):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        predictions = mask_rcnn(image_tensor)

    masks = predictions[0]['masks'].cpu().detach().numpy()
    scores = predictions[0]['scores'].cpu().detach().numpy()

    # Combine all masks that pass the confidence threshold
    final_mask = np.zeros((image.height, image.width), dtype=np.uint8)
    for i in range(len(masks)):
        if scores[i] > threshold:
            mask = (masks[i][0] > 0.5).astype(np.uint8) * 255
            final_mask = np.maximum(final_mask, mask)

    mask_image = Image.fromarray(final_mask)
    return image, mask_image

# Function to Apply Inpainting
def apply_inpainting(image, mask, prompt):
    inpainted_image = inpaint_pipe(
        prompt=prompt,
        image=image,
        mask_image=mask
    ).images[0]
    return inpainted_image

# User Input
image_path = "test_image.jpg"  # Replace with your image
prompt = input("Describe how you want the object changed: ")

# Process Image
original_image, mask_image = generate_mask(image_path)
edited_image = apply_inpainting(original_image, mask_image, prompt)

# Display Results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(original_image)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Generated Mask")
plt.imshow(mask_image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Modified Image")
plt.imshow(edited_image)
plt.axis("off")

plt.show()

# Save Output
edited_image.save("output_image.jpg")
print("Saved modified image as output_image.jpg")
