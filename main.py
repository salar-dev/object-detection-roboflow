import io
import cv2
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

# Load Image with PIL
img = cv2.imread("pepsi4.jpg")
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pilImage = Image.fromarray(image)

# Convert to JPEG Buffer
buffered = io.BytesIO()
pilImage.save(buffered, quality=100, format="JPEG")

# Build multipart form and post request
m = MultipartEncoder(fields={'file': ("imageToUpload", buffered.getvalue(), "image/jpeg")})

response = requests.post("https://detect.roboflow.com/pepsi-can-detection/1?api_key=GiGwZjBmOMekt0xkE95u", data=m, headers={'Content-Type': m.content_type})

data = response.json()

imWidth = int(data['predictions'][0]['width'])
imHeight = int(data['predictions'][0]['height'])
xp = int(data['predictions'][0]['x'])
yp = int(data['predictions'][0]['y'])
clasName = data['predictions'][0]['class']
confidence = float(data['predictions'][0]['confidence'])

start_x = xp - (imWidth / 2)
start_y = yp - (imHeight / 2)
start_point = (int(start_x), int(start_y))
end_x = xp + (imWidth / 2)
end_y = yp + (imHeight / 2)
end_point = (int(end_x), int(end_y))
fontScale = (imWidth * imHeight) / (1000 * 1000)

cv2.rectangle(img, start_point, end_point, color=(0, 255, 0), thickness=2)

# font
font = cv2.FONT_HERSHEY_SIMPLEX
# fontScale
fontScale = 1
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2

cv2.putText(img, clasName, (int(start_x), int(start_y) - 10), font, fontScale, color, thickness)

cv2.imshow('img', img)
cv2.waitKey(0)
