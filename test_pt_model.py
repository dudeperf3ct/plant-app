import torch
import cv2

THRESH = 0.70
model = torch.hub.load("ultralytics/yolov5", "custom", "models/yolov5s_weights.pt")
test_img_path = "test_images/test2.jpg"
img = cv2.imread(test_img_path)
results = model(test_img_path)

results.print()

df = results.pandas().xyxy[0]
print("-" * 80)
print("Before filtering ...")
print(df)
print("Unique classes detected: {}".format(df["name"].unique()))
print("Number of detections per unique class:\n{}".format(df["name"].value_counts()))
print("-" * 80)
print(f"Filtering detections using confidence threshold > {THRESH} ...")
filter_df = df[df["confidence"] > THRESH]
print(filter_df)
print("Unique classes detected: {}".format(filter_df["name"].unique()))
print(
    "Number of detections per unique class:\n{}".format(
        filter_df["name"].value_counts()
    )
)

for index, row in filter_df.iterrows():
    xmin, xmax, ymin, ymax = (
        int(row["xmin"]),
        int(row["xmax"]),
        int(row["ymin"]),
        int(row["ymax"]),
    )
    extract_img = img[ymin:ymax, xmin:xmax]
    cv2.imshow("detect", extract_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
