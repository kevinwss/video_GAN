import os
import imageio

def create_gif(dir_path, gif_name):  
  
    images = os.listdir(dir_path)
    frames = []  
    for image_name in images:  
        frames.append(imageio.imread(dir_path+ image_name))  
    # Save them as frames into a gif   
    imageio.mimsave(gif_name, frames, 'GIF', duration = 0.1)  
    
    return

create_gif("video/", "./gifs/test1.gif")
