#### TODO 1: Importing the Libraries
##"""

from flask import Flask
import socketio
import eventlet
from eventlet import wsgi
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

#@"""



#### TODO 5: Socket
##"""

sio = socketio.Server()


#### Des:
'''
# // sockets in general are used to perform real time communication between a client and a server,
# // when a client creates a single connection to a web socket server,
# /  it keeps listening for new events from the server allowing us to continuously update the client with data
# // we will be setting up a socket that is an io server and establish a bidirectional communication with the simulator
# // the server will then require a middleware to dispatch traffic to a socketio web application
# // we will combine a socket.io server with a flask webapp. 
'''

#@"""



#### TODO 2: Initialize the application
##"""

app = Flask(__name__)


### Des:
'''
# // this special variable __name__ will end up having the value of __main__ when executed. 
'''

#@"""



#### TODO 3: Trying the Flask Library by making a tiny website
"""

@app.route('/home')
def greeting():
    return "Welcome!"


### Des:
'''
# // specify a router decorator by the above syntax.
# // This route decorator will be used to tell flask what url we should use to trigger our function
# // should be above the function to associate with it.

# // this wasn't what we were aiming for, it was just an intro to flask.
'''

"""



#### TODO 8: Preprocesisng the Images
##"""

def img_preprocessing(img):
    img = img[60:135, :, :]  # // for center images

    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    img = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=0, sigmaY=0)

    img = cv2.resize(img, dsize=(200, 66) )

    img = img / 255

    return img

#@"""



#### TODO 7: Registering a Specific Event Handler
##"""

speed_limit = 13

@sio.on(event='telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image']) ) )
    image = np.asarray(image)
    image = img_preprocessing(image)
    image = np.array([image])
    steering_angle = float(nn_model_steering.predict(image) )
    throttle_value = 1.0 - (speed / speed_limit)
    print(f'{steering_angle}  {throttle_value}  {speed}')
    send_control(steering_angle=steering_angle, throttle=throttle_value)


### Des:
'''
# // What we are doing is listening for updates that will be sent to telemetry from the simulator
# /  and in the case of an event, the function is fired with the appropriate data.
# // As soon as the connection is established, we are setting the initial steering and throttle values
# /  and emitting them to the simulator, such that the car starts off as stationary, but then the simulator
# /  will send us back data which contains the current image of the frame, where the car is 
# /  presently located in the track and based on the image we want to run it through our model, 
# /  the model will extract the features from the image and predict the steering angle which we send back to 
# /  the simulation, and keep repeating that for the entire simulation such that
# /  the car starts driving on its own.
'''

'''
# // The image is base 64 encoded, so we have to decode it.
# // import base64
'''

'''
# // before we can open and identify the given image file with Image.open from the Python imaging
# /  library we need to use a buffer module to mimic our data like a normal file which we can further use
# /  for processing to do so, We make use of bytes IO
'''

'''
import image from the Python imaging library
'''

'''
# // The model expects a 4D arrays, whereas our images are only 3D, so we enclose the images by putting them
# /  inside of an array by np.array([])
'''

#@"""



#### TODO 4, 6: Running the tiny Flask website/ running the server with socket.io
##"""

## Event handler
@sio.on(event='connect')
def connect(sid, environ):
    print("Connected!")
    send_control(steering_angle=0, throttle=0)

def send_control(steering_angle, throttle):
    sio.emit(event='steer',
             data={"steering_angle": steering_angle.__str__(),
                   "throttle": throttle.__str__() } )



if __name__ == "__main__":
    # app.run(port=3000)  # // not needed anymore
    nn_model_steering = load_model('data 3\\3.CNN_modified_nvidia_model_CLR_images_aug_475_epochs_CLR.keras')
    app = socketio.Middleware(socketio_app=sio, wsgi_app=app)
    eventlet.wsgi.server(eventlet.listen(('', 4567) ), app)


### Des:
'''
# // 3000 is a port value which is a variable and not fixed. I can specify any other value.
# // to find the server, either search 'http://127.0.0.1:3000/home' whatever route I made'
# // or 'http://localhost:3000/home' whatever route I made'

## We then continue with the sio part.
# // Now, we can make use of a gate Web Server Gateway Interface (WSGI) to have our web server send any requests
# /  made by the client to the web application itself.
# // To launch this WSGI server, we simply create a socket, which what we already did, then the eventlet line.

# // first is the IP, we left as an empty string to listen on any available IP addresses
# // second is a specific port 4567, for the connection to work, we must listen on a specific port [4567]
# // third is the app variable from ths sio.Middleware func, the app is to which the requests are going to be sent.

# // Now, what we want is when there is a connection with the client we want to fire off an event handler.
# // generally you can call event names whatever you want but three names are reserved:
# /  connect, disconnect, message.
# // we will use connect, which fires upon a connection and will make use of the function.   
# // first the Session ID, second the environment (environ).
'''

## Lines: send control
'''
# // we are currently using a fixed value with no models or prediction just to test the connection.

'''

#@"""
