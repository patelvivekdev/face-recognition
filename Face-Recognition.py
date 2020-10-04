# Face Recognition with OpenCV

# Import Required Modules
# import OpenCV module
import cv2
# import os module for reading training data directories and paths
import os
#import numpy
import numpy as np

# There is no label 0 in our training data so subject name for index/label 0 is empty
subjects = ["", "MODIJI", "Shraddha Kapoor"]

# function to detect face using OpenCV
def detect_face(img):
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load OpenCV face detector, I am using LBP which is fast
    face_cascade = cv2.CascadeClassifier(
        'opencv-files/lbpcascade_frontalface.xml')

    # Detect multiscale images
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5)

    if (len(faces) == 0):
        return None, None
    # extract the face area
    (x, y, w, h) = faces[0]

    # return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]


# Read all persons' training images, detect face from each image
def prepare_training_data(data_folder_path):

    # Get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    # List to hold all subject faces
    faces = []
    # List to hold labels for all subjects
    labels = []

    # Each directory and read images within it
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue

        # Extract label number of subject from dir_name
        label = int(dir_name.replace("s", ""))

        # Sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # Get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

        # go through each image name, read image,
        # detect face and add face to list of faces
        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            # display an image window to show the image
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(500)

            # detect face
            face, rect = detect_face(image)

            # we will ignore faces that are not detected
            if face is not None:
                # add face to list of faces
                faces.append(face)
                # add label for this face
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


# Prepare our training data
print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

# print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

# create our LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()

# train our face recognizer of our training faces
face_recognizer.train(faces, np.array(labels))

# function to draw rectangle on image
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# function to draw text on give image
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

# Recognizes the person and draws a rectangle around
def predict(test_img):
    img = test_img.copy()
    # detect face from the image
    face, rect = detect_face(img)

    # predict the image using our face recognizer
    label, confidence = face_recognizer.predict(face)
    # get name of respective label returned by face recognizer
    label_text = subjects[label]

    # draw a rectangle around face detected
    draw_rectangle(img, rect)
    # draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1]-5)

    return img


print("Predicting images...")

# load test images
test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")

# perform a prediction
predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
print("Prediction complete")

# display both images
cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
