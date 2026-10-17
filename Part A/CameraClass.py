#use traitlets and widgets to display the image in Jupyter Notebook
import traitlets
from traitlets.config.configurable import SingletonConfigurable

#use opencv to covert the depth image to RGB image for displaying purpose
import cv2
import numpy as np

#using realsense to capture the color and depth image
import pyrealsense2 as rs

#multi-threading is used to capture the image in real time performance
import threading

class Camera(SingletonConfigurable):
    
    #this changing of this value will be captured by traitlets
    color_value = traitlets.Any()
    
    def __init__(self):
        super(Camera, self).__init__()
        
        #configure the color and depth sensor
        self.pipeline = rs.pipeline()
        self.configuration = rs.config()  
        
        #set resolution for the color camera
        self.color_width = 640
        self.color_height = 480
        self.color_fps = 30
        self.configuration.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.color_fps)

        #set resolution for the depth camera
        self.depth_width = 640
        self.depth_height = 480
        self.depth_fps = 30
        self.configuration.enable_stream(rs.stream.depth, self.depth_width, self.depth_height, rs.format.z16, self.depth_fps)

        #flag to control the thread
        self.thread_runnning_flag = False
        
        #start the RGBD sensor
        self.pipeline.start(self.configuration)
        self.pipeline_started = True
        frames = self.pipeline.wait_for_frames()

        #start capture the first color image
        color_frame = frames.get_color_frame()   
        image = np.asanyarray(color_frame.get_data())
        self.color_value = image

        #start capture the first depth image
        depth_frame = frames.get_depth_frame()           
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        self.depth_value = depth_colormap
        self.depth_frame = depth_image

    def _capture_frames(self):
        while(self.thread_runnning_flag==True): #continue until the thread_runnning_flag is set to be False
            frames = self.pipeline.wait_for_frames() #receive data from RGBD sensor
            
            color_frame = frames.get_color_frame() #get the color image
            image = np.asanyarray(color_frame.get_data()) #convert color image to numpy array
            self.color_value = image #assign the numpy array image to the color_value variable 

            depth_frame = frames.get_depth_frame() #get the depth image           
            depth_image = np.asanyarray(depth_frame.get_data()) #convert depth data to numpy array
            #conver depth data to BGR image for displaying purpose
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            self.depth_value = depth_colormap #assign the color BGR image to the depth value
            # add depth_image attribute
            self.depth_image = depth_image
            
    def is_object_ahead(self,distance_cm):
                
            #consider the central area of the vision sensor
            self.depth_image[:190,:]=0
            self.depth_image[290:,:]=0
            self.depth_image[:,:160]=0
            self.depth_image[:,480:]=0
            
            #For object avoidance, ignore distances that are lower than 100mm or bigger than 1000mm
            self.depth_image[self.depth_image<100]=0
            self.depth_image[self.depth_image>1000]=0            
            
            self.depth_image[0,0]=2000 # fix
            
            # true if object is within distance otherwise false
            if (self.depth_image[self.depth_image!=0].min()< (distance_cm *10)):
                return True
            else:
                return False
    
    def start(self): #start the data capture thread
        if self.thread_runnning_flag == False: #only process if no thread is running yet
            self.thread_runnning_flag=True #flag to control the operation of the _capture_frames function
            self.thread = threading.Thread(target=self._capture_frames) #link thread with the function
            self.thread.start() #start the thread

    def stop(self): #stop the data capture thread
        if self.thread_runnning_flag == True:
            self.thread_runnning_flag = False #exit the while loop in the _capture_frames
            self.thread.join() #wait the exiting of the thread       

def bgr8_to_jpeg(value):#convert numpy array to jpeg coded data for displaying 
    return bytes(cv2.imencode('.jpg',value)[1])