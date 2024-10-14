import cv2
import numpy as np

def calculate_transformation_matrix(max_z_end_effector, z_obj, robot_capturing_coord):
    values_tr00, values_tr01, values_tr10, values_tr11, values_off0, values_off1, values_H = np.load('./delta_manager/parameters/values.npy')
    p_H = np.poly1d(values_H)

    # total_height_robot = 86.7
    # H = total_height_robot - z_obj + robot_capturing_coord[2] 
    H = p_H(robot_capturing_coord[2]) - z_obj
    robot_capturing_coord_copy = robot_capturing_coord.copy()
    robot_capturing_coord_copy[2] = 0

    p00, p01, p10, p11 = np.poly1d(values_tr00), np.poly1d(values_tr01), np.poly1d(values_tr10), np.poly1d(values_tr11)
    tr_matrix = np.array([[p00(H), p01(H), 0], [p10(H), p11(H), 0], [0, 0, 0]])
    offp0, offp1 = np.poly1d(values_off0),  np.poly1d(values_off1)
    offset_matrix = np.array([offp0(H), offp1(H), max_z_end_effector + z_obj]) + robot_capturing_coord_copy

    return tr_matrix, offset_matrix


def pixel_to_robot_coordinates(pixel, z_obj=0, gripper='2f85', robot_capturing_coord=np.array([0,0,-37])):
    
    if gripper == '2f85':
        max_z_end_effector = -65.5
    elif gripper == 'Ehand':
        max_z_end_effector = -68
    elif gripper == 'pichgooshti':
        max_z_end_effector = -66
    else:
        raise Exception('Invalid gripper type')
        
    pixel_coord = [pixel[0],pixel[1],0]

    tr_matrix, offset_matrix = calculate_transformation_matrix(max_z_end_effector, z_obj, robot_capturing_coord)

    robot_coordinates = np.dot(pixel_coord, tr_matrix) + offset_matrix

    return robot_coordinates


def robot_coordinates_to_pixel(robot_coordinates, camera_height=50, z_obj=0, gripper='2f85', robot_capturing_coord=np.array([0,0,-37]), offset_valid=True):

    default_robot_height = 37
    if gripper == '2f85':
        max_z_end_effector = 65.5
    elif gripper == 'Ehand':
        max_z_end_effector = 68
    else:
        raise Exception('Invalid gripper type')
        

    if offset_valid:
        robot_homming_offset = np.load('./delta_manager/parameters/homming_offset.npy')
    else:
        robot_homming_offset = np.array([0,0,0])
    
    robot_homming_offset[2] = default_robot_height
    
    tr_height, offset_height = calculate_transformation_matrix(camera_height, max_z_end_effector, default_robot_height, z_obj, robot_capturing_coord, robot_homming_offset)

    pixel = np.dot(robot_coordinates - offset_height, np.linalg.pinv(tr_height))

    return [pixel[0], pixel[1]]


def undistort(frame):
    mtx = np.load("./delta_manager/parameters/camera_matrix.npy")
    dist = np.load("./delta_manager/parameters/dist_coeff.npy")
    newcameramtx = np.load("./delta_manager/parameters/newcameramtx.npy")
    roi = np.load("./delta_manager/parameters/roi.npy")
    
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # crop and save the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst
