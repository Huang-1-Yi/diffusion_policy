from third_party import co_tracker

class CoTrackerWrapper:
    def __init__(self, checkpoint):
        self.model = co_tracker.load_model(checkpoint)
        
    def track(self, video_frames, init_mask):
        # 视频帧格式: [T,H,W,3]
        return self.model(
            video_frames, 
            query_points=init_mask["keypoints"]
        )