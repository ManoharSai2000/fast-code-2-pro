import cv2

input = cv2.imread(
    "/mnt/c/Users/acer/Desktop/Courses/Spring24/FFII/Project/fast-code-2-pro/AlexNet/images/0/1.jpeg"
)
output = cv2.resize(input, (224, 224))
output = cv2.merge((output, output, output))

##save the image
cv2.imwrite(
    "/mnt/c/Users/acer/Desktop/Courses/Spring24/FFII/Project/fast-code-2-pro/AlexNet/images/0/1.jpeg",
    output,
)
