import tkinter as tk#for window
from collections import Counter #will automatically count the number of objects detected
import cv2 # computer vision library
from PIL import Image, ImageTk #image library
from ultralytics import YOLO #yolo algo, used for the image

# Load a pretrained model
model = YOLO("yolo26s-pose.pt")#change the letter after "11" to "n" for nano and weak device, "m" for medium and "l" for large and "x" for extra large and "s" for small, depending on your device's capabilities
# Run detection on an image
results = model("test2pos.jpg")

annotated_bgr = results[0].plot()  #Get the annotated image in BGR format
rgb_image = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

root =tk.Tk()
root.title("YOLO Object output detection") #output of the detection
 
pil_image = Image.fromarray(rgb_image)  #Convert to PIL Image
pil_image.thumbnail((900, 700))#resize image

#save the results to a file
results = model (r"C:\Users\ayton\OneDrive\Desktop\Py\pos.jpg", show=True)

#display in tkinter window
tk_image = ImageTk.PhotoImage(pil_image)
label = tk.Label(root, image=tk_image)
label.image = tk_image  #Keep a reference to avoid garbage collection
label.pack(padx=10, pady=10)

#counter for obj detected
counts = Counter([model.names[int(c)] for c in results[0].boxes.cls.tolist()])
summary = ", ".join(f"{k}: {v}" for k, v in counts.items()) or "No objects detected"
count_label = tk.Label(root, text=f"Detected objects: {summary}", font=("Arial", 14))
count_label.pack(padx=10, pady=10)

#print box details
for box in results [0].boxes:
    label = model.names[int(box.cls[0])]
    confidence = float(box.conf[0])
    coords = box.xyxy[0].tolist()  #Get the bounding box coordinates
    print(f"Detected {label} with confidence {confidence:.2f} at coordinates {coords}")
    
#loop
root.mainloop()