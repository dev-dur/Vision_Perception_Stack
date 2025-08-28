import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os
import math
import matplotlib.pyplot as plt
import time

def capture_webcam_img(IMAGE_PATH):
    if IMAGE_PATH and os.path.exists(IMAGE_PATH):
        img = cv2.imread(IMAGE_PATH)
       # Resize the image by scaling it by 10% (fx and fy are scaling factors)
        resized_img = cv2.resize(img, None, fx=0.5, fy=0.5)
        if img is None:
            print(f"Failed to load image: {IMAGE_PATH}")
            exit()
        IMG_HEIGHT_PX, IMG_WIDTH_PX = img.shape[:2]  # height, width
        print(f"Loaded image resolution: {IMG_WIDTH_PX} x {IMG_HEIGHT_PX}")
        return IMG_WIDTH_PX, IMG_HEIGHT_PX, resized_img
        
    else:
        cap = cv2.VideoCapture(0)  # Open webcam
        if not cap.isOpened():
            print("Unable to access webcam.")
            exit()
        IMG_WIDTH_PX = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        IMG_HEIGHT_PX = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Webcam resolution: {IMG_WIDTH_PX} x {IMG_HEIGHT_PX}")

        captured_image = None
        ret, captured_image = cap.read()
        print("Waiting for 5 seconds before capturing image...")
        time.sleep(3)
        cv2.imshow("Captured Image", captured_image)
        cv2.waitKey(3000)  # Show the captured image for 3 seconds
        cv2.destroyAllWindows()
        cap.release()
        
        if not ret:
            print("Failed to capture image.")
            cap.release()
            exit()
        
            
         
        return IMG_WIDTH_PX, IMG_HEIGHT_PX, captured_image

def estimate_depth_from_image(image, ordered_pts):
    # Load MiDaS model from torch.hub
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")  # or "DPT_Hybrid", "MiDaS_small"
    midas.eval()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)

    # Load transforms
    # MiDaS transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform if "DPT" in midas.__class__.__name__ else midas_transforms.small_transform

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    upscaled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(upscaled, -1, sharpen_kernel)

    # Convert to MiDaS input
    input_batch = transform(sharpened).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize depth map for visualization

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    equalized = clahe.apply(depth_map_normalized)
    depth_colored = cv2.applyColorMap(equalized, cv2.COLORMAP_MAGMA)

    # Step 2: Perspective Warp
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    ordered_pts = order_points(ordered_pts.astype(np.float32))
    width = int(max(
        np.linalg.norm(ordered_pts[0] - ordered_pts[1]),
        np.linalg.norm(ordered_pts[2] - ordered_pts[3])
    ))
    height = int(max(
        np.linalg.norm(ordered_pts[0] - ordered_pts[3]),
        np.linalg.norm(ordered_pts[1] - ordered_pts[2])
    ))

    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
    warped_depth = cv2.warpPerspective(depth_map, M, (width, height))
    warped_vis = cv2.normalize(warped_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- 1. Analyze full center column ---
    center_x = warped_depth.shape[1] // 2
    depth_column = warped_depth[:, center_x]  
    depth_column = cv2.GaussianBlur(depth_column[:, None], (5, 5), 0)[:, 0]

    # Compute depth differences
    background_depth = np.median(depth_column[-1:])  
    object_mask = np.abs(depth_column - background_depth) > (0.05 * background_depth)  # was 0.1

    object_indices = np.where(object_mask)[0]

    if len(object_indices) > 0:
        top_y = object_indices[0]
        bottom_y = object_indices[-1]
    else:
        top_y = 0
        bottom_y = len(depth_column) - 1

    # Calculate pixel height
    pixel_height_depth = abs(bottom_y - top_y)

    print(f"Pixel height depth: {pixel_height_depth} pixels")
    plt.figure(figsize=(8, 10))
    plt.imshow(warped_depth, cmap='magma', origin='lower')
    plt.colorbar(label='Depth Value')
    plt.scatter([center_x], [top_y], color='cyan', label='Top')
    plt.scatter([center_x], [bottom_y], color='lime', label='Bottom')
    plt.plot([center_x, center_x], [top_y, bottom_y], color='white', linewidth=2)
    plt.title('Detected Top and Bottom')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(True)
    plt.show()

    return depth_colored, warped_vis, pixel_height_depth

def run_yolo_obb_on_image(resized_image, model_path, margin=1.0):
    # Load model and image
    model = YOLO(model_path)
    orig = resized_image
    
    assert orig is not None, f"Failed to load {orig}"
    image = orig.copy()

    # Run OBB inference
    results = model(image, conf=0.2, task="obb")[0]
    if results.obb is None or len(results.obb.xyxyxyxy) == 0:
        print("No objects detected!")
        return None, None, None, None, None        

    for poly, conf, cls in zip(
            results.obb.xyxyxyxy.cpu().numpy(),
            results.obb.conf.cpu().numpy(),
            results.obb.cls.cpu().numpy().astype(int)):

        # original 4-corner points
        pts = poly.reshape((4, 2)).astype(np.float32)

        x_coords = pts[:, 0]
        y_coords = pts[:, 1]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        


        # Perspective warp
        def order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect

        ordered_pts = np.array([
                [x_min, y_min],  # Top-left
                [x_max, y_min],  # Top-right
                [x_max, y_max],  # Bottom-right
                [x_min, y_max]   # Bottom-left
            ], dtype=np.float32)
        # The center is the midpoint of the bounding box after expansion
        center_x = int((x_min + x_max) / 2)
        center_y = int((y_min + y_max) / 2)
        center_pt = (center_x, center_y)

        # Compute dimensions
        width = int(max(
            np.linalg.norm(ordered_pts[0] - ordered_pts[1]),
            np.linalg.norm(ordered_pts[2] - ordered_pts[3])
        ))
        height = int(max(
            np.linalg.norm(ordered_pts[0] - ordered_pts[3]),
            np.linalg.norm(ordered_pts[1] - ordered_pts[2])
        ))

        y_coords = [pt[1] for pt in ordered_pts]
        top_y = min(y_coords)
        bottom_y = max(y_coords)
        top_point = ordered_pts[y_coords.index(top_y)]
        bottom_point = ordered_pts[y_coords.index(bottom_y)]

        pixel_height_rgb  = int(y_max - y_min)
        print(f"height rgb:{pixel_height_rgb}")


        dst_pts = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        # Draw fixed box
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        cv2.putText(image, f"{model.names[cls]} {conf:.2f}", (int(x_min), int(y_min) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
        # Warp
        M = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
        if M is None or not isinstance(M, np.ndarray):
            print("Error: M is not a valid matrix. Skipping depth estimation.")
            return
        warped = cv2.warpPerspective(image, M, (width, pixel_height_rgb ))

        # Put pixel height text
    #     cv2.putText(warped, f"Height: {pixel_height_rgb }px", (10, 30),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # cv2.imshow("Detected Objects", image)
    # #print(f"M matrix:\n{M}")
    # cv2.imshow('Warped Image', warped)
    

    
    return top_point, bottom_point, ordered_pts, pixel_height_rgb , center_pt, image 

def calculate_height(resized_img, pixel_height_rgb, top_pt_warped, bottom_pt_warped, pixel_height_depth, IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX):
        # --- Camera Parameters ---
        CAMERA_HEIGHT_CM = 77     # Real camera height from ground
        CAMERA_PITCH_DEG = -30     
        FOCAL_LENGTH_MM = 3.6     # Focal length in millimeters (120)
        SENSOR_WIDTH_MM = 3.6     # Sensor width in millimeters (example value, please adjust according to your camera specs)
        H_REF = 60      
        FOV_deg = 65        
        FOCAL_PX = (FOCAL_LENGTH_MM * IMAGE_WIDTH_PX) / SENSOR_WIDTH_MM
        FOCAL_LENGTH_PX = (FOCAL_PX) / np.tan(np.radians(FOV_deg))
        print(f"focal px: {FOCAL_LENGTH_PX}")


        py = pixel_height_rgb 
        p1 = (int(top_pt_warped[0]), int(top_pt_warped[1]))
        p2 = (int(bottom_pt_warped[0]), int(bottom_pt_warped[1]))


        v_beta1 = (FOCAL_LENGTH_PX / p1[1]+ 1e-6) 
        v_beta2 = (FOCAL_LENGTH_PX / p2[1]+ 1e-6)
        v_beta = (FOCAL_LENGTH_PX / py)
        v0 = int(IMAGE_HEIGHT_PX // 2)
        print(f"image height: {IMAGE_HEIGHT_PX}")

        
        # Final vanishing point f
        vanish = v0 - FOCAL_LENGTH_PX * np.tan(np.radians(CAMERA_PITCH_DEG))
        # If your camera was level (0° pitch), the horizon/vanish point would be around v0 = 640px (middle of 1280px image).
        
        # Scaling factor gamma_beta for pitch compensation if the ground is uneven
        #f = v_beta * np.sin(np.radians(CAMERA_PITCH_DEG)) + FOCAL_LENGTH_PX * np.cos(np.radians(CAMERA_PITCH_DEG))
        gamma_beta = (v_beta * np.sin(np.radians(abs(v_beta1-v_beta2)))) + vanish * np.cos(np.radians(abs(v_beta1-v_beta2)))
        print(f"gemma: {gamma_beta}")
        f = abs(vanish / gamma_beta) 
        
        print(f"vanish point: {vanish}")
        horizon_x = IMAGE_WIDTH_PX // 2
        horizon_y = int(max(0, min(vanish, IMAGE_HEIGHT_PX)))
        
        Hop = py 
        Hc = CAMERA_HEIGHT_CM
        Hcp = abs(py - f)
        
        # Ho = ((Hc / Hcp) * Hop) 
        Ho = ((Hc / Hop) * Hcp ) 
        print(f"actual height: {Ho} cm")

        # Calculate the error in depth (in terms of pixels) between p1 and p2
        depth_height_cm = (pixel_height_depth * (H_REF / Hop)) 
        print(f"depth in cm: {depth_height_cm} cm")
        # depth_error = abs(pixel_height_depth - pixel_height_rgb)
        # print(f"depth error: {depth_error}")

        # If Ho is smaller than the depth error, apply the scaling
        if Ho > H_REF:
            scale_factor = Ho/H_REF
            final_height = depth_height_cm * scale_factor
        else:
            scale_factor = depth_height_cm/H_REF
            final_height = Ho * scale_factor

        print(f"scaling factor: {scale_factor}")
        print(f"final: {final_height} cm")
        # --- Error estimation ---
        try:
            error_px = (f ** 2 * abs(v_beta1-v_beta2)) / max((gamma_beta - abs(v_beta1-v_beta2)), 1e-6)  # avoid divide by zero
            percentage_error = (abs((Ho - depth_height_cm))/Ho) * 100
            print(f"%error: {percentage_error}")   
            # print(f"pixel error: {error_px}")   
        except Exception as e:
            error_px = 0

     
        # Draw horizon
        result_img = resized_img.copy()
        vanish_clamped = int(max(0, min(vanish, result_img.shape[0] - 1)))
        center_x = result_img.shape[1] // 2
        cv2.circle(result_img, (center_x, vanish_clamped), 20, (0, 0, 255), -1)
        cv2.line(result_img, (0, vanish_clamped), (IMAGE_WIDTH_PX, vanish_clamped), (255, 0, 0), 2)
        

        # # # --- Display Height Measurements ---
        # # cv2.putText(result_img, f"Final Height: {final_height:.2f} cm", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # # cv2.putText(result_img, f"Error: {percentage_error:.2f}%", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # --- Show the result ---
        cv2.imshow("Result Visualization", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        return result_img, f, horizon_y, horizon_x, Ho, Hop, FOCAL_LENGTH_PX

def compute_angle(f, horizon_y, horizon_x, center_pt, image, IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX):
    # Camera center (bottom middle of the image)
    camera_pt = (IMAGE_WIDTH_PX // 2, IMAGE_HEIGHT_PX)

    # 0-degree reference (straight upward)
    zero_deg_pt = (IMAGE_WIDTH_PX // 2 + 5, 0)


    # Object center
    object_pt = (int(center_pt[0]), int(center_pt[1]))


    # 3. Compute vectors
    ref_vector = np.array([zero_deg_pt[0] - camera_pt[0], zero_deg_pt[1] - camera_pt[1]], dtype=float)
    obj_vector = np.array([object_pt[0]  - camera_pt[0], object_pt[1]  - camera_pt[1]], dtype=float)

    # Normalize
    ref_vector_norm = ref_vector / np.linalg.norm(ref_vector)
    obj_vector_norm = obj_vector / np.linalg.norm(obj_vector)

    # Dot product for angle
    dot = np.clip(np.dot(ref_vector_norm, obj_vector_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    angle_deg = np.degrees(angle_rad)

    # Cross product to determine direction (left or right)
    cross = np.cross(ref_vector_norm, obj_vector_norm)
    if cross > 0:
        angle_deg = -angle_deg  # Left side = negative

    # Clamp between [-30°, +30°]
    angle_deg = float(np.clip(angle_deg, -30, 30))

    print(f"Computed angle: {angle_deg:.2f}°")


    # cv2.line(image, camera_pt, zero_deg_pt, (0, 255, 0), 2)
    # cv2.line(image, camera_pt, object_pt, (0, 0, 255), 2)
    # cv2.putText(image, f"Angle: {angle_deg:.2f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)



    # cv2.imshow("Angle Visualization", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return angle_deg



def get_distance(depth_colored, actual_height, angle_deg, pixel_height, focal_length_px):  
    # Calculate the distance using the horizontal angle and scaling factor
    distance = abs(actual_height / np.tan(60)) 
    distance = distance/10
    # Using scaling factor for accurate estimation
    print(f"distance: {distance}")
    
    return distance


def save_measurements_to_file(filepath, Ho, distance_cm, angle_deg):

    with open(filepath, 'w') as f:
        f.write(f"Actual height (Ho): {Ho:.2f} cm\n")
        f.write(f"Distance       : {distance_cm:.2f} cm\n")
        f.write(f"Angle          : {angle_deg:.2f}°\n")
    print(f"[+] Results saved to {filepath}")
