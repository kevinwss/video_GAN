"""
The input pipeline in this file takes care of loading video datasets.
Videos are stored as JPEG files of horizontally stacked frames.

The pipeline takes care of normalizing, cropping and making all videos
the same frame length.
Videos are randomized and put into batches in a multi-threaded fashion.
"""

import tensorflow as tf
import os


class InputPipeline(object):
    def __init__(self, root_dir, index_file, read_threads, batch_size,
                 num_epochs=1, video_frames=32, reshape_size=64):
        """
        :param root_dir: root directory containing the index_file and all the videos
        :param index_file: list of video paths relative to root_dir
        :param read_threads: number of threads used for parallel reading
        :param batch_size: size of the batches to output
        :param num_epochs: number of epochs, use None to make infinite
        :param video_frames: number of frames every video should have in the end
                             if a video is shorter than this repeat the last frame
        :param reshape_size: videos frames are stored as 126x126 images, reshape them to
                             reduce the dimensionality
        """
        self.read_threads = read_threads
        self.batch_size = batch_size
        self.video_frames = video_frames
        self.reshape_size = reshape_size

        with open(os.path.join(root_dir, index_file)) as f:
            content = f.readlines()
        if "index" in index_file:  #index.txt
            content = [root_dir + '/' + x.strip() for x in content]
        else:
            content = [x.strip() for x in content]
        self._filename_queue = tf.train.string_input_producer(content, shuffle=True, num_epochs=num_epochs)

    def __read_video(self):
        """
        read one video of the filename queue and return it a image (JPG)
        of horizontally stacked frames
        """
        file_reader = tf.WholeFileReader()
        _, image_data = file_reader.read(self._filename_queue)
        image = tf.cast(tf.image.decode_jpeg(image_data, channels=3), tf.float32)
        print("image",image.shape)
        return image
        
        
    def process_img(self,video):
        # process to image
        shape = tf.shape(video)
        length = 128
        # print(shape) # original shape of the frames of a video. One video corresspond to one big frame.
        #image = video[:128,:128,:]  # only use the first frame to train
        image = video
        image =  tf.image.resize_images(image, [self.reshape_size, self.reshape_size])

        return tf.subtract(tf.div(image, 127.5), 1.0)
        
    def __preprocess(self, video):
        """
        takes a image of horizontally stacked video frames and transforms
        it to a tensor of shape:
        [self.video_frames x self.reshape_size x self.reshape_size x 3]
        """
        shape = tf.shape(video)
        print("shape",shape)
        
        frames = tf.reshape(video, [-1, shape[1], shape[1], 3])

        # resize the image
        frames = tf.image.resize_images(frames, [self.reshape_size, self.reshape_size])
        
        # make videos of unique frame size, if the video is too short repeat
        # the last frame, if it is too long cut it.
        def extend_length():
            last = frames[-1]
            end_frames = tf.tile(tf.expand_dims(last, 0), [self.video_frames - tf.shape(frames)[0], 1, 1, 1])
            return tf.concat([frames, end_frames], 0)

        def keep_length(): return frames[0:self.video_frames]

        frames = tf.cond(tf.less(tf.shape(frames)[0], self.video_frames), extend_length, keep_length)
        print("frames",frames.shape)
        return tf.subtract(tf.div(frames, 127.5), 1.0)

    def input_pipeline(self):
        #vid_list = [self.__preprocess(self.__read_video()) for _ in range(self.read_threads)]
        
        vid_list = [self.process_img(self.__read_video()) for _ in range(self.read_threads)]
        print("vid_list",len(vid_list))
        print(vid_list)
        video_batch = tf.train.batch([vid_list], batch_size=self.batch_size,
                                     # TODO(Bernhard): check if read_threads here actually speeds things up
                                     num_threads=self.read_threads, capacity=self.batch_size * 4, enqueue_many=True,
                                     shapes=[self.reshape_size, self.reshape_size, 3]
                                     #shapes=[self.video_frames, self.reshape_size, self.reshape_size, 3]
                                     )
        print("video_batch",video_batch.shape)  
        print(video_batch)                      
        return video_batch
