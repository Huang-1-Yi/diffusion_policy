from configs import params
from utils.parallel import create_pipeline

def main():
    # 初始化流水线
    pipeline = create_pipeline(
        modules=[
            ("capture", "CameraNode", params.CAMERA_CONFIG),
            ("segment", "SoMWrapper", params.SOM_CONFIG),
            ("vlm", "VLMBinder", params.VLM_CONFIG),
            ("track", "CoTrackerWrapper", params.TRACK_CONFIG),
            ("policy", "GuidedDiffusion", params.POLICY_CONFIG)
        ],
        queue_sizes={
            "raw_frames": 10,
            "masks": 5,
            "target_feats": 1,
            "tracks": 20
        }
    )
    
    # 启动异步执行
    pipeline.start()
    
    # 监控线程
    while pipeline.is_alive():
        print(f"当前状态: {pipeline.status_report()}")
        time.sleep(1)

if __name__ == "__main__":
    main()