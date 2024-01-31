from fastai.vision.all import *

# Load the learner
learn = load_learner('concrete_classifier.pkl')

# Function to create a PILImage object
def get_image(image_path):
    return PILImage.create(image_path)

# Example usage:
image_path = 'images/concrete6.jpg'
img = get_image(image_path)

# Predict the image
pred, pred_idx, probs = learn.predict(img)
print(f'Prediction: {pred}; Probability: {probs[pred_idx]}')