import cv2

def detect_objects_orb(image_path, template_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(image, None)
    keypoints2, descriptors2 = orb.detectAndCompute(template, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

# Example usage:
image_path = "image.jpg"
template_path = "template.jpg"
matches = detect_objects_orb(image_path, template_path)
print("Detected object matches:", matches)
