from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

# Load model
model = load_model("groundingdino/config/GroundingDINO_SwinB_cfg.py", "weights/groundingdino_swinb_cogcoor.pth")

# Test image and parameters
IMAGE_PATH = ".asset/cat_dog.jpeg"  # Use sample image from repository
TEXT_PROMPT = "dog . cat . chair ."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# Load and process image
image_source, image = load_image(IMAGE_PATH)
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

# Create and save annotated image
annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
cv2.imwrite("test_output.jpg", annotated_frame)