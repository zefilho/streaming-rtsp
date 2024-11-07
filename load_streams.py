
import glob
import math
import os
import re
import time
import cv2
import numpy as np
import requests
import torch
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from threading import Thread, Lock
from urllib.parse import urlparse
from queue import Queue
from typing import List, Tuple, Union
import logging


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LoadStreams:
    """
        Stream Loader for various types of video streams.
        Supports RTSP, RTMP, HTTP, and TCP streams, and YouTube.
    """

    def __init__(self, sources: Union[str, List[str]] = "file.streams",
                 vid_stride: int = 1, buffer: bool = False,
                 max_buffer_size: int = 10):
        """
            Initialize stream loader for multiple video sources.

            Args:
                sources (str or List[str]): The input file or URLs for video streams.
                vid_stride (int): Frame rate stride for processing every n-th frame.
                buffer (bool): Whether to buffer frames.
                max_buffer_size (int): Maximum number of frames to buffer.
        """
        self.buffer = buffer
        self.running = True
        self.vid_stride = vid_stride
        self.max_buffer_size = max_buffer_size
        self.lock = Lock()

        #Load sources from list file or input list
        sources = Path(sources).read_text().splitlines() if os.path.isfile(sources) else [sources]
        self.bs = len(sources)
        self.sources = [LoadStreams.__clean_str(s) for s in sources]

        self.bs = len(sources)
        self.caps = []
        self.threads = []
        self.img_buffers = [Queue(maxsize=max_buffer_size) for _ in range(self.bs)]
        self.shapes = []
        self.fps = []

        # Initialize streams
        for i, s in enumerate(sources):
            cap, shape, fps = self.init_stream(s, i)
            self.caps.append(cap)
            self.shapes.append(shape)
            self.fps.append(fps)

            thread = Thread(target=self.update_stream, args=(i, cap), daemon=True)
            thread.start()
            self.threads.append(thread)

    def init_stream(self, source: str, index: int) -> Tuple[cv2.VideoCapture, Tuple[int, int, int], float]:
        """Initialize a single video stream."""
        LOGGER.info(f"Initializing stream {index + 1}/{self.bs}: {source}...")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ConnectionError(f"Failed to open stream: {source}")

        # Retrieve stream properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30  # Fallback to 30 FPS if undefined
        frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or float("inf")

        LOGGER.info(f"Stream {index + 1}: {width}x{height} at {fps:.2f} FPS with {frames_count} frames.")
        return cap, (height, width, 3), fps

    def update_stream(self, index: int, cap: cv2.VideoCapture):
        """Thread to update image buffers with frames from the video stream."""
        while self.running and cap.isOpened():
            if not self.img_buffers[index].full():
                success, frame = cap.read()
                if success:
                    with self.lock:
                        self.img_buffers[index].put(frame)
                else:
                    LOGGER.warning(f"Stream {index} unresponsive. Attempting to reopen...")
                    cap.open(self.sources[index])
            else:
                time.sleep(0.01)  # Sleep to avoid busy-waiting

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        """Return the next batch of frames from each stream."""
        self.count += 1
        frames = []
        for i, buffer in enumerate(self.img_buffers):
            try:
                frame = buffer.get(timeout=1 / self.fps[i])
            except:
                frame = np.zeros(self.shapes[i], dtype=np.uint8)  # Fallback if no frame is available
            frames.append(frame)
        return self.sources, frames

    def close(self):
        """Close all video streams and stop threads."""
        self.running = False
        for thread in self.threads:
            thread.join(timeout=5)
        for cap in self.caps:
            cap.release()
        cv2.destroyAllWindows()


    @staticmethod
    def __clean_str(s):
        """
        Cleans a string by replacing special characters with '_' character.

        Args:
            s (str): a string needing special characters replaced

        Returns:
            (str): a string with special characters replaced by an underscore _
        """
        return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)
