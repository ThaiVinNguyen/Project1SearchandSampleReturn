# Project: Search and Sample Return

The goals / steps of this project are the following:

- Training / Calibration:
  - Download the simulator and take data in "Training Mode"
  - Test out the functions in the Jupyter Notebook provided
  - Add functions to detect obstacles and samples of interest (golden rocks)
  - Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
  - Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

- Autonomous Navigation / Mapping
  -  Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook).
  - Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands.
  - Iterate on your perception and decision function until your rover does a reasonable job of navigating and mapping. (Mapping 40% of the map, at 60% Fidelity, and locate at least 1 rock sample).

[//]: # (Image References)

[image1]: ./calibration_images/calibration_threshold.jpg
[image2]: ./misc/terrain_path.jpg
[image3]: ./misc/rover_mapping.jpg 
[image4]: ./misc/my_rover_mapping.jpg 
[image5]: ./misc/unity_res.jpg
[image6]: ./misc/autonomous_mode.jpg 

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points
---
### - Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

##### - Here it is

### - Notebook Analysis

#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

![alt text][image1]

###### **Figure**  **1** : Calibration data plots with all 3 thresholds

![alt text][image2]

###### **Figure**  **2** : Computer vision plots results from notebook

#### 2. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result.
##### A. Did a few modifications to different functions. All details are shown in the jupyter notebook 'Rover_Project_Test_Notebook.ipynb'

![alt text][image3]

###### **Figure**  **3** : Output Image results with test_dataset provided

![alt text][image4]

###### **Figure**  **4** :Output Image results with my recorded test data (attach in My_dataset file)

### - Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

##### A. perception.py modifications:

###### a. These color_thresh function modifications make it so that it outputs all 3 thresholds, one for navigable path, rock samples, and rock samples; respectively. Thought, red and green thresholds higher than 100 and blue threshold lower than 50 do the trick to recognize the yellow pixels from the rock samples.
```python
def color_thresh(img, rgb_thresh=(160, 160, 160,100,100,50)):
    # Create an array of zeros same xy size as img, but single channel
    color_select_path = np.zeros_like(img[:,:,0])
    color_select_rock = np.zeros_like(img[:,:,0])
    color_select_obstacle = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    # Threshold for terrain path
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Threshold for rocks
    between_thresh = (img[:,:,0] > rgb_thresh[3] ) \
                & (img[:,:,1] > rgb_thresh[4] ) \
                & (img[:,:,2] < rgb_thresh[5] )
    # Threshold for obstacles
    below_thresh = (img[:,:,0] < rgb_thresh[0]) \
                & (img[:,:,1] < rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select_path[above_thresh] = 1
    color_select_rock[between_thresh] = 1
    color_select_obstacle[below_thresh] = 1
    # Return the binary image
    return color_select_path, color_select_rock, color_select_obstacle
```
###### b. Changes to the perspect_transform function generate a second output to consider the outside field of view, which would otherwise be considered part of the obstacles without this method.
```python
    def perspect_transform(img, src, dst):      
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
        mask = cv2.warpPerspective(np.ones_like(warped[:,:,0]), M, (img.shape[1], img.shape[0])) #create a mask keep same size as input image
        return warped,mask
```    
###### c. Changes to the perception_step() function have been made to apply all the functions previously stated to provide a complete computer vision image that can be used to tell the rover where it can travel and where it can't, as well as adding the ability to detect rock samples.

####### - I also applied the to_polar_coords() function to the rock 'x' and 'y' pixels to provide the rover distance and direction to where the rock samples are, for steering guidance.
```python
    def perception_step(Rover):
        # Perform perception steps to update Rover()
        # TODO: 
        # NOTE: camera image is coming to you in Rover.img
        # 1) Define source and destination points for perspective transform
        xpos, ypos = Rover.pos
        yaw = Rover.yaw
        dst_size=8
        bottom_offset=6
        world_size = Rover.worldmap.shape[0]
        scale = 2* dst_size
        source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
        destination = np.float32([[Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - bottom_offset],
                        [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - bottom_offset],
                        [Rover.img.shape[1]/2 + dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset], 
                        [Rover.img.shape[1]/2 - dst_size, Rover.img.shape[0] - 2*dst_size - bottom_offset],
                        ])
        # 2) Apply perspective transform
        warped, mask = perspect_transform(Rover.img, source, destination)
        # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
        threshed_path, threshed_rock, threshed_obstacle = color_thresh(warped)
        threshed_obs = np.absolute(np.float32(threshed_obstacle))*mask
        # 4) Update Rover.vision_image (this will be displayed on left side of screen)
            # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
            #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
            #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
        Rover.vision_image[:,:,0] = threshed_obs * 255 #thresholded binary image - obstacles color
        Rover.vision_image[:,:,1] = threshed_rock * 255 #thresholded binary image - rocks color
        Rover.vision_image[:,:,2] = threshed_path * 255 #thresholded binary images - terrain color

        # 5) Convert map image pixel values to rover-centric coords
        xpix, ypix = rover_coords(threshed_path)
        obs_xpix, obs_ypix = rover_coords(threshed_obs)
        rock_xpix, rock_ypix = rover_coords(threshed_rock)
        # 6) Convert rover-centric pixel values to world coordinates
        x_world, y_world = pix_to_world(xpix,ypix,xpos,ypos,yaw,world_size,scale)
        obs_xworld, obs_yworld = pix_to_world(obs_xpix,obs_ypix,xpos,ypos,yaw,world_size,scale)
        rock_xworld, rock_yworld = pix_to_world(rock_xpix,rock_ypix,xpos,ypos,yaw,world_size,scale)
        # 7) Update Rover worldmap (to be displayed on right side of screen)

            # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
            #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
            #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
        Part_obs = Rover.worldmap[:,:,2]>0
        Rover.worldmap[Part_obs, 0] = 0
        Rover.worldmap[rock_yworld, rock_xworld, 1] = 255
        Rover.worldmap[y_world, x_world, 2] = 255
        Rover.worldmap[obs_yworld, obs_xworld, 0] = 255
        # 8) Convert rover-centric pixel positions to polar coordinates
        rover_dist, rover_angle = to_polar_coords(xpix, ypix)
        rock_dist, rock_angle = to_polar_coords(rock_xpix, rock_ypix)
        # Update Rover pixel distances and angles
            # Rover.nav_dists = rover_centric_pixel_distances
            # Rover.nav_angles = rover_centric_angles
        Rover.nav_dists = rover_dist
        Rover.nav_angles = rover_angle
        Rover.rock_dists = rock_dist
        Rover.rock_angles = rock_angle



        return Rover
```
##### B. decision.py modifications:

###### a. Made these changes to the decision_step() function to provide the extra capability to locate and steer towards rock samples when found, stop when near a sample, and pickup sample when it has stopped in front of the rock sample.
```python
def decision_step(Rover):

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        # Check for Rover.mode status
        if len(Rover.rock_angles) != 0:
            Rover.sample_pos_found = Rover.rock_angles
            Rover.steer = np.clip(np.mean(Rover.rock_angles * 180/np.pi), -15, 15)
            if len(Rover.rock_angles) >= 20:
                Rover.sample_pos_found = Rover.rock_dists
                if Rover.vel < 1:
                # Set throttle value to throttle setting
                    Rover.throttle = 0.1
                    Rover.brake = 0
                elif Rover.vel >= 1:
                    Rover.brake = 5
                    Rover.throttle = 0
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
            elif len(Rover.rock_angles) <= 20:
                Rover.sample_pos_found = len(Rover.rock_angles)
                # Set mode to "stop" and hit the brakes!
                #Rover.throttle = 0
                if Rover.vel < 0.7:
                # Set throttle value to throttle setting
                    Rover.throttle = 0.1
                    Rover.brake = 0
                elif Rover.vel >= 0.7:
                    Rover.throttle = 0
                    Rover.brake = 5
                else: # Else coast
                    Rover.throttle = 0              
                # Set brake to stored brake value
                if Rover.near_sample:
                    Rover.throttle = 0
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
                        Rover.send_pickup = True
        elif Rover.mode == 'forward' and len(Rover.rock_angles) == 0 and Rover.near_sample == 0: 
            Rover.sample_pos_found = len(Rover.rock_angles)
            # Check the extent of navigable terrain
            if len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set

                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                # If there's a lack of navigable terrain pixels then go to 'stop' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                # Set mode to "stop" and hit the brakes!
                Rover.throttle = 0
                # Set brake to stored brake value
                Rover.brake = Rover.brake_set
                Rover.steer = 0
                Rover.mode = 'stop'

            # If we're already in "stop" mode then make different decisions
        elif Rover.mode == 'stop' and len(Rover.rock_angles) == 0:
            # If we're in stop mode but still moving keep braking
            if Rover.vel > 0.2:
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
                Rover.steer = 0
            # If we're not moving (vel < 0.2) then do something else
            elif Rover.vel <= 0.2:
                # Now we're stopped and we have vision data to see if there's a path forward
                if len(Rover.nav_angles) < Rover.go_forward:
                    Rover.throttle = 0
                    # Release the brake to allow turning
                    Rover.brake = 0
                    # Turn range is +/- 15 degrees, when stopped the next line will induce 4-wheel turning
                    Rover.steer = -25 # Could be more clever here about which way to turn
                # If we're stopped but see sufficient navigable terrain in front then go!
                elif len(Rover.nav_angles) >= Rover.go_forward:
                    # Set throttle back to stored value
                    Rover.throttle = Rover.throttle_set
                    # Release the brake
                    Rover.brake = 0
                    # Set steer to mean angle
                    Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
                    Rover.mode = 'forward' 

    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
    
    return Rover
```
##### C. drive_rover.py modifications:

###### a. Made these changes to the __init__() function to provide the extra variables to the rover for storing rock sample distance and angles, along with a string variable that's used to prompt in different situations for testing and debugging purposes.
```python
class RoverState():
    def __init__(self):
        self.start_time = None # To record the start time of navigation
        self.total_time = None # To record total duration of naviagation
        self.img = None # Current camera image
        self.pos = None # Current position (x, y)
        self.yaw = None # Current yaw angle
        self.pitch = None # Current pitch angle
        self.roll = None # Current roll angle
        self.vel = None # Current velocity
        self.steer = 0 # Current steering angle
        self.throttle = 0 # Current throttle value
        self.brake = 0 # Current brake value
        self.nav_angles = None # Angles of navigable terrain pixels
        self.nav_dists = None # Distances of navigable terrain pixels
        self.ground_truth = ground_truth_3d # Ground truth worldmap
        self.mode = 'forward' # Current mode (can be forward or stop)
        self.throttle_set = 0.2 # Throttle setting when accelerating
        self.brake_set = 10 # Brake setting when braking
        # The stop_forward and go_forward fields below represent total count
        # of navigable terrain pixels.  This is a very crude form of knowing
        # when you can keep going and when you should stop.  Feel free to
        # get creative in adding new fields or modifying these!
        self.stop_forward = 50 # Threshold to initiate stopping
        self.go_forward = 500 # Threshold to go forward again
        self.max_vel = 2 # Maximum velocity (meters/second)
        # Image output from perception step
        # Update this image to display your intermediate analysis steps
        # on screen in autonomous mode
        self.vision_image = np.zeros((160, 320, 3), dtype=np.float) 
        # Worldmap
        # Update this image with the positions of navigable terrain
        # obstacles and rock samples
        self.worldmap = np.zeros((200, 200, 3), dtype=np.float) 
        self.samples_pos = None # To store the actual sample positions
        self.samples_to_find = 0 # To store the initial count of samples
        self.samples_located = 0 # To store number of samples located on map
        self.samples_collected = 0 # To count the number of samples collected
        self.near_sample = 0 # Will be set to telemetry value data["near_sample"]
        self.picking_up = 0 # Will be set to telemetry value data["picking_up"]
        self.send_pickup = False # Set to True to trigger rock pickup
        self.rock_dists = None #Distances of rock pixels
        self.rock_angles = None #Angles of rock pixels
        self.sample_pos_found = None # to print string of sample pos situation
        self.mode = 'stop' # Current mode (can be forward or stop)
```
#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.

##### In autonomous mode, I managed to map at least 40% with at least 60% fidelity. Also, some capability to detect, navigate towards rock samples, and pick them up was added. There are instances where it would crash against the rocks in the middle of the map, sometimes getting stuck. Also, sometimes it stays in a loop going to the same places over and over, or just stays around in circles looking for a greater navigable path.

##### Below is the simulator setup used for the Autonomous Mode.

![alt text][image5]

Figure 5: Resolution Setup for Simulation

###### Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

![alt text][image6]

Figure 6: Result which run Simulator for autonomous mode

### - Here I will talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.


##### The Rover data set up in the drive_rover.py; from color_thresh() to pix_to_world(). Modified slightly the color_thresh() function to output the 3 threshold required to detect obstacles, path, and rock samples. I also added directional data to Rover for rock samples by applying to_polar_coords() to rock pixels.

##### In the decision_step() function, I added simple `if` and `elif` chain events to trigger stopping and slower speeds upon detecting rock samples, with some steering towards the rock pixel angles provided by the perception step (a little rough around the edges but did the trick most of the time). Occasionally, it stops on top of the rock samples and gets stuck. The rover sometimes, stays running around in circles.

##### Would improve time optimization to have the rover run and stop more efficiently. Furthermore, make the rover go back to where it started; can also prevent the rover from running around in circles.
