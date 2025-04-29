from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(model=r'best3.pt')
    results = model.predict(source=r'ultralytics/assets/qwe.jpg',
                            save=True,
                            show=True,
                            show_labels=False,#不展示标签
                            save_conf=True,
                            show_conf=False, #展示置信度
                            show_boxes=True,
                            save_txt=True,
                            line_width=6 ,#调整线宽
                            )
    for r in results:
        print(f"Detected {len(r)} objects in image")
