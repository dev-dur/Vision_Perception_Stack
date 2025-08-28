import cv2
from utilis import *

def main():
    # Load an image for depth estimation and object detection E:/OneDrive/FYP/custom model/main prog/test_image (2).jpeg
    IMAGE_PATH = ""
    if IMAGE_PATH is None:
        print(f"Failed to load image: {IMAGE_PATH}")
        exit()

    MODEL_PATH = "E:/OneDrive/FYP/custom model/model training/runs/obb/train/weights/best.pt"

    # image width and height calculations ( IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX, )
    IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX,resized_img = capture_webcam_img(IMAGE_PATH)

    # object detection -> height in pixels  (pixel_height, top_pt_warped, bottom_pt_warped, M = )
    top_point, bottom_point, ordered_pts, pixel_height_rgb , center_pt, image  = run_yolo_obb_on_image(resized_img, MODEL_PATH, margin=1) 

    # # depth estimation -> depth height in pixels
    depth_colored, _, pixel_height_depth = estimate_depth_from_image(image, ordered_pts) 

    # height calculations
    _, f, horizon_y, horizon_x, Ho, Hop, FOCAL_LENGTH = calculate_height(resized_img, pixel_height_rgb, top_point, bottom_point, pixel_height_depth, IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX)
    
    # calcualte angle
    angle_deg = compute_angle(f, horizon_y, horizon_x, center_pt, image, IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX)
    distance_m = get_distance(depth_colored, Ho, angle_deg, Hop, FOCAL_LENGTH)
    save_measurements_to_file('measurements.txt', Ho, distance_m, angle_deg)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()