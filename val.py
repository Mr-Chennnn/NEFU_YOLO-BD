import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('best3.pt')  # 选择训练好的权重路径
    print(model)
    per_fps = 0
    count = 1  # 计算10次的fps均值
    for i in range(count):
        metrics = model.val(data='ultralytics/cfg/datasets/data.yaml',
                            split='test',  # split可以选择train、val、test 根据自己的数据集情况来选择.
                            imgsz=640,
                            batch=1,
                            show_labels=False,

                            iou=0.4,
                            # rect=False,
                            # save_json=True, # if you need to cal coco metrice
                            # project='runs/val',
                            # name='exp',
                            )
        speed_metrics = metrics.speed
        total_time = sum(speed_metrics.values())
        fps = 1000 / total_time
        per_fps += fps


    print(f"FPS: {per_fps/count}")
