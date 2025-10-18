import numpy as np
import cv2


def scale_features_from_sift(keypoints):
    """
    Extract unique scale features from SIFT keypoints.

    A single coordinate (x, y) can have multiple keypoints with the same scale and up to four different orientations.
    This function extracts one tuple (x, y, scale) for each unique coordinate.
    
    Args:
        keypoints: List of cv2.KeyPoint objects from SIFT detection
        
    Returns:
        scale_features: Array of shape (N, 3) containing (x, y, scale) for each unique coordinate
    """
    unique_xys = {}
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        key = (x, y)
        if key not in unique_xys:
            unique_xys[key] = kp
    
    scale_features = np.array([[kp.pt[0], kp.pt[1], kp.size] for kp in unique_xys.values()])
    return scale_features


def orientation_features_from_sift(keypoints):
    """
    Extract orientation features from SIFT keypoints.
    
    This function extracts (x, y, angle) tuples from all keypoints with valid angles along with their sizes.
    Valid angles are those that are not -1. Angles are converted to radians.
    
    Args:
        keypoints: List of cv2.KeyPoint objects from SIFT detection
        
    Returns:
        orientation_features: Array of shape (N, 3) containing (x, y, angle) for each keypoint
        sizes: Array of shape (N,) containing the size for each keypoint
    """
    orientation_features = []
    sizes = []
    for kp in keypoints:
        if kp.angle != -1:
            orientation_features.append([kp.pt[0], kp.pt[1], np.deg2rad(kp.angle)])
            sizes.append(0.5 *kp.size)
    return np.array(orientation_features), np.array(sizes)


def draw_scale_features(img, scale_features, color, thickness=2):
    """
    Draw scale features as circles on the input image.
    Circles are drawn at the scale feature's coordinates, with a radius equal to the scale feature's scale.
    
    Args:
        img: Input image
        scale_features: Array of shape (N, 3) containing (x, y, scale)
        color: Circle color (B, G, R)
        thickness: Circle thickness
    """
    for feat in scale_features:
        x, y, scale = int(feat[0]), int(feat[1]), feat[2]
        radius = int(scale / 2)
        cv2.circle(img, (x, y), radius, color, thickness, lineType=cv2.LINE_AA)

def draw_orientation_features(img, orientation_features, sizes, color, thickness=2):
    """
    Draw orientation features as lines on the input image.
    
    Lines are drawn from the orientation feature's coordinates in the direction of the feature's angle, with a length equal to the feature's size.
    
    Args:
        img: Input image
        orientation_features: Array of shape (N, 3) containing (x, y, angle)
        sizes: Array or scalar containing the size for each feature
        color: Line color (B, G, R)
        thickness: Line thickness
    """
    for i, feat in enumerate(orientation_features):
        x, y, angle = feat[0], feat[1], feat[2]
        length = sizes[i] if hasattr(sizes, '__iter__') else sizes
        
        end_x = int(x + length * np.cos(angle))
        end_y = int(y + length * np.sin(angle))
        
        cv2.line(img, (int(x), int(y)), (end_x, end_y), color, thickness, lineType=cv2.LINE_AA)
        # cv2.circle(img, (int(x), int(y)), int(length), color, thickness, lineType=cv2.LINE_AA)


def perspective_warp(img, H, border_mode=cv2.BORDER_CONSTANT, border_value=(255, 255, 255)):
    """
    Apply perspective warp to image with automatic output size calculation.
    
    Args:
        img: Input image
        H: 3x3 homography matrix
        border_mode: Border extrapolation mode (default: cv2.BORDER_CONSTANT)
        border_value: Border value for cv2.BORDER_CONSTANT mode (default: (255, 255, 255))
        
    Returns:
        warped_img: Warped image
        H_translated: Translated homography matrix
        (min_x, min_y): Minimum coordinates after warping
    """
    h, w = img.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
    warped_corners = H @ corners
    warped_corners = warped_corners[:2] / warped_corners[2]
    
    min_x, min_y = warped_corners.min(axis=1)
    max_x, max_y = warped_corners.max(axis=1)
    
    output_size = (int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y)))
    translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    
    H_translated = translation @ H
    warped_img = cv2.warpPerspective(img, H_translated, output_size, 
                                     borderMode=border_mode, borderValue=border_value)
    
    return warped_img, H_translated, (min_x, min_y)

