import cv2
import numpy as np

def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Captured Image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('captured_image.jpg', frame)
    else:
        print("Error: Unable to capture image.")

def remove_small_pixels(image):
    # Apply any specific preprocessing or remove small pixels here
    return image

def break_pixels(image):
    # Apply any specific operations to break pixels here
    return image

def connect_gray_scales(image):
    # Apply any specific operations to connect grayscale pixels here
    return image

def threshold_image(image):
    # Apply thresholding to the image
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return thresh

def component_labeling(image):
    # Apply connected component analysis
    num_labels, labels = cv2.connectedComponents(image)
    color_map = np.uint8(np.random.uniform(0, 255, size=(num_labels, 3)))
    labeled_image = color_map[labels]
    labeled_image[labels == 0] = [0, 0, 0]
    return labeled_image

def centroids(image):
    # Find contours in the image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw centroids on the image
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(image, (cX, cY), 5, (0, 255, 0), -1)
            cv2.putText(image, f"({cX}, {cY})", (cX - 50, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

def count_shapes(image):
    # Apply any specific operations to count shapes
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Shapes Counted Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def live_counting():
    cap = cv2.VideoCapture('input_video.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_video.avi', fourcc, 30.0, (640, 480))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        cv2.imshow('Live Counting', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    operation = input("Enter the operation you want to perform: \n1. Remove Small Pixels\n2. Break Pixels\n3. Connect Gray Scales\n4. Threshold Image\n5. Component Labeling\n6. Centroids\n7. Count Shapes in Image\n8. Live Counting (for video)\n")
    
    if operation == '1':
        capture_image()
    elif operation == '1':
        # Load an image
        image_path = input("Enter the path to the image file: ")
        image = cv2.imread(image_path)
        if image is not None:
            # Remove small pixels
            processed_image = remove_small_pixels(image)
            cv2.imshow('Removed Small Pixels', processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Image file not found.")
    elif operation == '2':
        # Load an image
        image_path = input("Enter the path to the image file: ")
        image = cv2.imread(image_path)
        if image is not None:
            # Break pixels
            processed_image = break_pixels(image)
            cv2.imshow('Broken Pixels', processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Image file not found.")
    elif operation == '3':
        # Load an image
        image_path = input("Enter the path to the image file: ")
        image = cv2.imread(image_path)
        if image is not None:
            # Connect grayscale pixels
            processed_image = connect_gray_scales(image)
            cv2.imshow('Connected Gray Scales', processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Image file not found.")
    elif operation == '4':
        # Load an image
        image_path = input("Enter the path to the image file: ")
        image = cv2.imread(image_path)
        if image is not None:
            # Threshold image
            processed_image = threshold_image(image)
            cv2.imshow('Thresholded Image', processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Image file not found.")
    elif operation == '5':
        # Load an image
        image_path = input("Enter the path to the image file: ")
        image = cv2.imread(image_path)
        if image is not None:
            # Component labeling
            processed_image = component_labeling(image)
            cv2.imshow('Labeled Components', processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Image file not found.")
    elif operation == '6':
        # Load an image
        image_path = input("Enter the path to the image file: ")
        image = cv2.imread(image_path)
        if image is not None:
            # Centroids
            processed_image = centroids(image)
            cv2.imshow('Centroids', processed_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Error: Image file not found.")
    elif operation == '7':
        # Load an image
        image_path = input("Enter the path to the image file: ")
        image = cv2.imread(image_path)
        if image is not None:
            # Count shapes in image
            count_shapes(image)
        else:
            print("Error: Image file not found.")
    elif operation == '8':
        live_counting()
    else:
        print("Invalid operation.")

def show_logo():
    print(r" (                 (                 )  (    ")
    print(r" )\ )          (   )\ )    (      ( /(  )\ ) ")
    print(r"(()/(    (   ( )\ (()/(    )\     )\())(()/( ")
    print(r"/(_))   )\  )((_) /(_))((((_)(  ((_)\  /(_)) ")
    print(r"(_))  _ ((_)((_)_ (_))   )\ _ )\  _((_)(_))   ")
    print(r"|_ _|| | | | | _ )|_ _|  (_)_\(_)| \| |/ __|  ")
    print(r" | | | |_| | | _ \ | |    / _ \  | .` |\__ \  ")
    print(r"|___| \___/  |___/|___|  /_/ \_\ |_|\_||___/  ")

if __name__ == "__main__":
    show_logo()
    main()
