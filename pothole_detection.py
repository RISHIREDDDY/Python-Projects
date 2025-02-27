import cv2
import numpy as np
import pygame
import time
import geocoder
from matplotlib import pyplot as plt
from geopy.geocoders import Nominatim
import tkinter as tk  # Import Tkinter


# Function to get user's location
def get_location():
    geolocator = Nominatim(user_agent="pothole-detection")
    location = geolocator.geocode("Hyderabad, India")
    if location:
        print(f"Latitude: {location.latitude}, Longitude: {location.longitude}")
        return location.latitude, location.longitude
    else:
        print("Location not found")
        return None, None


# Fetch the user's location
lat, lon = get_location()

# Load the image
im = cv2.imread("C:/Users/HP/Downloads/index4.jpg")
if im is None:
    print("Error loading image.")
    exit()

# Convert to grayscale
gray1 = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
cv2.imwrite('graypothholeresult.jpg', gray1)

# Contour detection code
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours1, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours2, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img2 = im.copy()
cv2.drawContours(img2, contours2, -1, (250, 250, 250), 1)

# Display the user's location on the image
if lat and lon:
    location_text = f"Location: {lat:.5f}, {lon:.5f}"
    cv2.putText(img2, location_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow('Contour Detection', img2)
cv2.waitKey(0)

# Read another image and process
img = cv2.imread('C:/Users/HP/Downloads/index2.jpg', 0)
if img is None:
    print("Error loading image.")
    exit()

ret, thresh = cv2.threshold(img, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, 1, 2)

if contours:
    cnt = contours[0]
    perimeter = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)
    epsilon = 0.1 * perimeter
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    print(f"Perimeter: {perimeter}")
    print(f"Area: {area}")

    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 100 or rect[3] < 100:
            continue
        x, y, w, h = rect
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 8)
        cv2.putText(img2, 'Object Detected', (x + w + 40, y + h), 0, 2.0, (0, 255, 0))

    cv2.imshow("Detected Objects", img2)
    cv2.waitKey(0)

k = cv2.isContourConvex(cnt)
print(f"Convexity: {k}")

# Image processing steps
blur = cv2.blur(im, (5, 5))
gblur = cv2.GaussianBlur(im, (5, 5), 0)
median = cv2.medianBlur(im, 5)

kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(median, kernel, iterations=1)
dilation = cv2.dilate(erosion, kernel, iterations=5)
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
edges = cv2.Canny(dilation, 9, 220)

# Plot the images in subplots
plt.subplot(331), plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(332), plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)), plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(333), plt.imshow(cv2.cvtColor(gblur, cv2.COLOR_BGR2RGB)), plt.title('Gaussian Blur')
plt.xticks([]), plt.yticks([])
plt.subplot(334), plt.imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB)), plt.title('Median Blur')
plt.xticks([]), plt.yticks([])
plt.subplot(335), plt.imshow(cv2.cvtColor(erosion, cv2.COLOR_BGR2RGB)), plt.title('Erosion')
plt.xticks([]), plt.yticks([])
plt.subplot(336), plt.imshow(cv2.cvtColor(closing, cv2.COLOR_BGR2RGB)), plt.title('Closing')
plt.xticks([]), plt.yticks([])
plt.subplot(337), plt.imshow(img, cmap='gray'), plt.title('Dilated Image'), plt.xticks([]), plt.yticks([])
plt.subplot(338), plt.imshow(edges, cmap='gray'), plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()

# Alert the driver using sound
pygame.init()
pygame.mixer.music.load("C:/Users/HP/Downloads/buzz.mp3")
pygame.mixer.music.play()
time.sleep(5)
pygame.mixer.music.stop()


# Function to display alert message with theme
def show_alert():
    alert_window = tk.Tk()
    alert_window.title("Alert")

    # Set window size
    alert_window.geometry("300x150")  # Width x Height
    alert_window.configure(bg="#f8d7da")  # Light red background

    # Custom font styles
    title_font = ("Helvetica", 16, "bold")
    message_font = ("Helvetica", 12)

    # Create a title label
    alert_label = tk.Label(alert_window, text="Pothole Detected!",
                           font=title_font, bg="#f8d7da", fg="#721c24", padx=20, pady=20)
    alert_label.pack()

    # Create a message label
    message_label = tk.Label(alert_window, text="Drive Safely!",
                             font=message_font, bg="#f8d7da", fg="#721c24")
    message_label.pack(pady=5)

    # Create a button to close the alert
    alert_button = tk.Button(alert_window, text="OK", command=alert_window.destroy,
                             padx=10, pady=5, bg="#c3e6cb", fg="#155724", font=message_font)
    alert_button.pack(pady=10)

    # Run the main loop
    alert_window.mainloop()


# Show alert after the sound
show_alert()

# Close all OpenCV windows
cv2.destroyAllWindows()
