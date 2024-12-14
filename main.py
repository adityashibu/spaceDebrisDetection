from ultralytics import YOLO
model = YOLO("yolo11n.pt")

if __name__ == '__main__':
    model.train(data=r"C:\Users\Aditya Uni\spaceDebrisDetection\data.yaml", epochs=50, batch=16, imgsz=640)

    results = model.val(data=r"C:\Users\Aditya Uni\spaceDebrisDetection\data.yaml")
    print(results)

    # Run inference on a test image or directory
    results = model.predict(source=r"C:\Users\Aditya Uni\Downloads\test.jpg", imgsz=640)

    # Results will contain predictions, including bounding boxes, class names, and confidence scores
    for result in results:
        result.show()  # To display predictions
        result.save()  # To save predictions``