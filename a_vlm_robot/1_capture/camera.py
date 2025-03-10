import cv2
from multiprocessing import shared_memory

class CameraNode:
    def __init__(self, config):
        self.cam_id = config["camera"]["id"]
        self.shm = shared_memory.SharedMemory(
            name=config["shared_mem"]["name"],
            size=config["shared_mem"]["size"]
        )
        
    def stream_loop(self):
        cap = cv2.VideoCapture(self.cam_id)
        while True:
            ret, frame = cap.read()
            if ret:
                # 将帧数据写入共享内存
                self.shm.buf[:] = frame.flatten()
                
    def get_frame(self):
        return np.frombuffer(self.shm.buf, dtype=np.uint8).reshape(
            self.config["camera"]["height"],
            self.config["camera"]["width"], 3
        )