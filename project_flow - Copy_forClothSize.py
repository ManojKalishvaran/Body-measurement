import cv2 as cv
import numpy as np
import mediapipe as mp
import warnings 
import matplotlib.pyplot as plt
import rembg
warnings.simplefilter('ignore')

"""Step 1 - load & process"""
#Load and preprocess img :
img = cv.imread(r"D:/Body measurement project/images/aswa.jpg")
k = min(1.0, 1024/max(img.shape[0], img.shape[1]))

img = cv.resize(img, None, fx=k, fy=k, interpolation=cv.INTER_LANCZOS4)
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


count_fall_back = 0
"""step 2 - coordinate points"""
#Human pose estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils
print(img.shape)
height, width, _ = img.shape

def get_xy_for_points(img, land_mark_no):
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    coordinates = pose.process(rgb)
    height, width, _ = img.shape
    if coordinates.pose_landmarks:
        x, y = int(coordinates.pose_landmarks.landmark[land_mark_no].x*width), int(coordinates.pose_landmarks.landmark[land_mark_no].y*height)
    return x, y

def length_line(img, point_1, point_2):
    height, width, _ = img.shape
    x_point_1, y_point_1 = get_xy_for_points(img, point_1)
    x_point_2, y_point_2 = get_xy_for_points(img, point_2)
    
    length = np.sqrt((x_point_2-x_point_1)**2 + (y_point_2-y_point_1)**2)
    return length

def length_in_cm(img, point_1, point_2):
    lenght = length_line(img, point_1, point_2)
    return lenght * ratio

def length_line_pixel_incm(x,y):
    leng = np.sqrt((y[0]-x[0])**2 + (y[1]-x[1])**2)
    length = leng*ratio
    return np.round(length, decimals=2)

name = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'K', 'L', 'M','N']
point = [12, 11, 24, 23, 28, 27, 0, 13, 15,28, 32]

point_number = {k:v for k, v in zip(name, point)}
print(point_number['F'])
print(point_number)

"""step 3 - ratio between cm and pixel"""
# x_nose, y_nose = get_xy_for_points(img, 0)
# x_ankle, y_ankle = get_xy_for_points(img, 27)

# height_per = int(input('Enter length of your arm in c.m :'))
height_per = 29
ratio = height_per/length_line(img, 12, 14)
print(ratio)

"""step 4 - chest and waist"""

def mid_point(x1, y1, x2, y2):
    mid_x, mid_y = int((x1+x2)/2), int((y1+y2)/2)
    return mid_x, mid_y

#center point of line CD (I):
C_x, C_y = C = get_xy_for_points(img, point_number['C'])
D_x, D_y = D = get_xy_for_points(img, point_number['D'])



CD_mid = I = mid_point(C_x, C_y, D_x, D_y)

#center point of line AB (H):
A_x, A_y = A = get_xy_for_points(img, point_number['A'])
B_x, B_y = B = get_xy_for_points(img, point_number['B'])
AB_mid = H = mid_point(A_x, A_y, B_x, B_y)
K = get_xy_for_points(img, point_number['K'])
L = get_xy_for_points(img, point_number['L'])
M = get_xy_for_points(img, point_number['M'])
E = get_xy_for_points(img, point_number['E'])
N = get_xy_for_points(img, point_number['N'])
F = get_xy_for_points(img, point_number['F'])
print(E)

#J - 0.25 of HI line for chest:
H_x, H_y = H
I_x, I_y = I
HI_mid = (int((H_x+I_x)/2), int((H_y+I_y)/2))
W = mid_point(HI_mid[0], HI_mid[1], I[0], I[1])
HI_quat = J = mid_point(H[0], H[1], HI_mid[0], HI_mid[1])


# cv.circle(img, H, 3, (0, 0, 0), 2)
# cv.circle(img, J, 3, (0, 0, 0), 2)
# cv.circle(img, I, 3, (0, 0, 0), 2)




#Canny edge to find radius R:
def canny_edge_detector(image, low_threshold, high_threshold):
    blurred_image = cv.medianBlur(image, (3), 0)
    edge_map = cv.Canny(image, low_threshold, high_threshold)
    return edge_map

low_threshold = 60
high_threshold = 100
edge_map = canny_edge_detector(img, low_threshold, high_threshold)
edge_map_fin = np.where(edge_map == 255, 1, 0)
# print(edge_map_fin)
plt.axis('off')
plt.imshow(edge_map_fin, cmap='gray')
plt.show()
print(np.unique(edge_map_fin))

def sobel_edge_detector(image):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image = cv.medianBlur(gray, (9))

    # Apply Sobel edge detection
    sobel_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)  # Horizontal edges
    sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)  # Vertical edges

    # Convert the results back to uint8
    sobel_x = cv.convertScaleAbs(sobel_x)
    sobel_y = cv.convertScaleAbs(sobel_y)

    # Combine the two edges
    sobel_combined_ori = cv.bitwise_or(sobel_x, sobel_y)
    sobel_combined = np.where(sobel_combined_ori >130, 1, 0)
    return sobel_combined

sobel_edge = sobel_edge_detector(img)
plt.imshow(sobel_edge, cmap='gray')
plt.show()

def remove_bg(img):
    output = rembg.remove(img)
    alphachannel = output[:, :, 3]
    binary_mask = np.where(alphachannel>0, 0, 1)
    return binary_mask

# move horizontally to find radius of waist:
mid_BD = mid_point(B_x, B_y, D_x, D_y)    

def no_pixel_for_centi(nu):
    px_for_1cm = 1/ratio
    final_val_ad = px_for_1cm*nu
    return int(final_val_ad)


def move_through_canny(edge_map, J, mid_BD, search_ad, fin_ad, val_push_right=0, val_pull_left=0):
    width = edge_map.shape[1]  
    move = J[0]
    add_px_to_check = no_pixel_for_centi(search_ad)  
    final_add = no_pixel_for_centi(fin_ad)
    min_push_from_border = no_pixel_for_centi(val_push_right)
    val_pull_from_border = no_pixel_for_centi(val_pull_left)
    
    for i in range(J[0], width):
        if J[0] - 1 < move < mid_BD[0] + add_px_to_check:
            if mid_BD[0]+min_push_from_border-val_pull_from_border < move < mid_BD[0] + add_px_to_check and int(edge_map[J[1]][move]) == 1:
                print('Edge found at:', (move, J[1]))
                return (move, J[1])
                val_cnt = 0
            else:
                move += 1  
        else:
            print('No valid edge found, returning fallback')
            return (int(np.round(mid_BD[0] + final_add)), J[1])
            val_cnt = 1
    return (move, J[1])

bg_remd_body = remove_bg(img)


"""CHEST"""


def get_radius(H, J, ratio):
    return (np.sqrt((H[0]-J[0])**2+(H[1]-J[1])**2))*ratio

def get_circumference(radius):
    return np.round((2*3* radius), decimals=2)

def get_intercept(first_point, second_point):
    a,b = first_point[0]
    c,d = first_point[1]
    e,f = second_point[0]
    g,h = second_point[1]
    m = (f-h)/(e-g)
    
    y = b = d
    x = (g*m-(h-y))/m
    return (int(x), int(y))
    
right_point_chest =CR= (B[0], J[1])
mid_point_chest = CM = get_intercept(list([J,(B[0], J[1])]), list([D,B]))
print(mid_point_chest, "\n")

chest_point = move_through_canny(edge_map_fin, J, mid_point_chest,   5, 1.32, val_push_right=1)

dif_sh_x, dif_sh_y =  chest_point[0]-J[0], chest_point[1]-J[1]
chest_point_left = (J[0]-dif_sh_x, J[1]-dif_sh_y)
# chest_radius =get_radius(J, chest_point, ratio)
# Chest_circumference = get_circumference(chest_radius)
chest_length = length_line_pixel_incm(chest_point_left, chest_point)
chest_radius = length_line_pixel_incm(J, chest_point)
print('chest_radius', chest_radius)
cv.line(img, chest_point_left, chest_point,(0, 28, 255), 2)
cv.putText(img, str(chest_length )+ " cm", J,  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
print("length of the chest is :", chest_length , "cm \n")

"""Waist"""
"""upper"""
# Move from I in horizontally :
WR = (B[0],W[1])
mid_BD_waist = get_intercept(list([W,WR]), list([D,B]))

mid_BD_waist =WM = get_intercept([I, WR], [D,B])

waist_point_upper = move_through_canny(bg_remd_body, W, mid_BD_waist,  10, 5.9, val_push_right=3)
# waist_radius = get_radius(W, waist_point_upper, ratio)
# waist_circumference = get_circumference(waist_radius)

df_up_x, df_up_y = waist_point_upper[0]-W[0], waist_point_upper[1]-W[1]
left_waist_point_up = W[0]-df_up_x, W[1]-df_up_y
length_waist_up = (length_line_pixel_incm(left_waist_point_up, waist_point_upper))
cv.putText(img, str(length_waist_up )+ " cm", W, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
cv.line(img, left_waist_point_up, waist_point_upper, (255, 36, 0), 2)
print(f'Upper waist length is : {length_waist_up} cm \n' )

"""lower"""
waist_point_lower = move_through_canny(bg_remd_body, I, D, 15, 11, val_push_right = 4)
# lower_waist_raidus = get_radius(I, waist_point_lower, ratio)
# waist_circumference_lower = get_circumference(lower_waist_raidus)

df_down_x, df_down_y = waist_point_lower[0]-I[0], waist_point_lower[1]-I[1]
left_waist_point_lower = I[0]-df_down_x, I[1]-df_down_y
length_waist_lower = (length_line_pixel_incm(left_waist_point_lower, waist_point_lower))
if length_waist_lower<length_waist_up - 2:
    length_waist_lower = length_waist_up + 3 
    cv.putText(img,  str(length_waist_lower )+ " cm", I, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    cv.line(img, (left_waist_point_up[0]-int(no_pixel_for_centi(1.5)),left_waist_point_lower[1]), ((waist_point_upper[0]+int(no_pixel_for_centi(1.5))) ,waist_point_lower[1]), (255, 165, 0), 2)
    print(f'Lower waist length is : {length_waist_lower} cm ...returned using upper length')
else:
    cv.putText(img,  str(length_waist_lower )+ " cm", I, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    cv.line(img, left_waist_point_lower, waist_point_lower, (255, 165, 0), 2)
    print(f'Lower waist length is : {length_waist_lower} cm \n')


"""Step - 5 Shoulder """
shoulder_point = move_through_canny(edge_map_fin, H, B, 4, 2, val_push_right=0.5)
shoulder_length = length_line_pixel_incm(H, shoulder_point) *2
print(f"shoulder length is :{shoulder_length} cm \n")
dif_x, dif_y = shoulder_point[0]-H[0], shoulder_point[1]-H[1]

cv.putText(img, str(shoulder_length )+ " cm", H, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
shoulder_point_left = (H[0]-dif_x, H[1]-dif_y)
cv.line(img, shoulder_point_left, shoulder_point,(26, 255, 50), 2)
mid_of_shoulder = mid_point(H[0], H[1], shoulder_point[0], shoulder_point[1])
# cv.imshow('img', img)
# cv.waitKey(0)

"""Step - 6 body length"""

def move_vertically_through_canny_up(edge_map, center_shoulder, search_ad, fin_ad):
    starting_point = center_shoulder[1] 
    move = starting_point
    search_range = starting_point - no_pixel_for_centi(search_ad)
    final_val_to = no_pixel_for_centi(fin_ad)

    while move > search_range:
        if edge_map[move][center_shoulder[0]] == 1:  
            print("Edge found at: ", (center_shoulder[0], move))
            val_cnt = 1
            return center_shoulder[0], move
        else:
            move -= 1
    
    print("Returning fallback value")
    return center_shoulder[0], int(starting_point - final_val_to)


def move_vertically_through_canny_down(edge_map, center_waist, search_ad, fin_ad, push_down=0):
    starting_point = center_waist[1]  # y-coordinate
    move = starting_point + no_pixel_for_centi(push_down)
    search_range = starting_point + no_pixel_for_centi(search_ad)
    final_val_to = no_pixel_for_centi(fin_ad)

    height = edge_map.shape[0]  # Get the height of the image

    while move < min(search_range, height):
        if edge_map[move][center_waist[0]] == 1:  # x-coordinate is fixed, move y-coordinate
            print("Edge found at: ", (center_waist[0], move))
            return int(center_waist[0]), move
        else:
            move += 1
    
    print("Returning fallback value")
    return center_waist[0], int(starting_point + final_val_to)

waist_cent = (mid_of_shoulder[0], W[1])
up_body_points = move_vertically_through_canny_up(sobel_edge, mid_of_shoulder , 15, 10)
lower_body_points = move_vertically_through_canny_down(sobel_edge, waist_cent, 25, 10)
if lower_body_points[0]<waist_point_lower[0]:
    lower_body_points = (up_body_points[0], waist_point_lower[1]+ no_pixel_for_centi(2))
    print('\n body length is less than lower waist...')
else:
    lower_body_points = lower_body_points
body_length = length_line_pixel_incm(up_body_points, lower_body_points)
print(f'lower body points {lower_body_points}')
print(f"Body length is {body_length} cm \n")
cv.putText(img, str(body_length )+ " cm", up_body_points, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
cv.line(img, up_body_points, lower_body_points, (0, 255, 255), 2)


"""step - 6 hand length """
BK = length_line_pixel_incm(B, K)
KL = length_line_pixel_incm(K, L)
hand_length = BL = (BK+KL).round(2)
print(f"hand length is {BL} cm \n")
cv.line(img, B, K, (255, 0, 255), 2)
cv.line(img, K, L, (255, 0, 255), 2)
cv.putText(img, str(BL )+ " cm", K, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)


"""step - 7 pant length """
def inseam_length(I, F):
    inseam_start = move_vertically_through_canny_down(sobel_edge, I, 25, 6,push_down=15)
    inseam_length = length_line_pixel_incm(inseam_start, F)
    return inseam_start, inseam_length
    

WI = length_line_pixel_incm(W, I)
LL = (I[0], F[1])
inseam_point , inseam_len = inseam_length(I, LL)

ILL = length_line_pixel_incm(I, F)
pant_length = np.round(WI+ILL, decimals=2)
print(f"pant length is {pant_length} cm \n")
print(f"inseam length is {inseam_len} cm \n")

cv.line(img, inseam_point,F, (245, 0, 200), 2 )
cv.line(img, WM, F,(238, 10, 150), 2)
cv.putText(img, str(inseam_len )+ " cm", inseam_point, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

cv.putText(img, str(pant_length )+ " cm", M, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)


"""estimate size"""

import math


def convert_cm_inch(size):
    size_inch = size/2.54
    return size_inch

def half_cir(radius):
    half_cir = np.pi*radius
    return half_cir
 
def length_to_circumference(length):
    return math.pi * length  

def cm_to_inches(cm):
    return cm / 2.54


def get_size_shirt(measured):
    chest_m, waist_m, sleeves_m, shoulder_m, length_m = measured
    size_chart = [
        (39, 39, 24, 17.5, 28.5, "XS Size (Extra Small)"),
        (41, 41, 24.5, 18, 29, "S Size (Small)"),
        (42.5, 42.5, 25, 18.5, 29.5, "M Size (Medium)"),
        (44, 44, 25.5, 18.75, 29.75, "M Size (Medium)"),
        (45.5, 45.5, 26, 19, 30, "L Size (Large)"),
        (47, 47, 26.5, 19.5, 30.5, "L Size (Large)"),
        (48.5, 48.5, 27, 20, 31, "XL Size (Extra Large)"),
        (50, 50, 27, 20.5, 31.25, "XL Size (Extra Large)"),
        (52.5, 52.5, 27.25, 21, 31.5, "XXL Size (Double Extra Large)"),
        (55, 55, 27.25, 21.5, 32.25, "XXXL Size (Triple Extra Large)"),
        (58, 58, 27.5, 22, 33, "XXXXL Size (4 Extra Large)")
    ]
    
    size_ret = None
    min_dif = float("inf")

    def find_diff(size_tuple, measured):
        diff = sum(abs(size_tuple[i] - measured[i]) for i in range(5))  # Compare all 5 measurements
        return diff

    for size in size_chart:
        dif = find_diff(size, measured)
        if dif < min_dif:
            min_dif = dif
            size_ret = size[5]  # Return the size label

    return size_ret if size_ret else "No matching size"


chest = cm_to_inches(get_circumference(chest_radius))
waist = cm_to_inches(get_circumference(length_waist_up/2))
sleeve = cm_to_inches(hand_length)
shoulder = cm_to_inches(shoulder_length)
length = cm_to_inches(body_length)

shirt_size = get_size_shirt((chest, waist, sleeve, shoulder, length))
print(f"Shirt size is: {shirt_size}")


def get_pants_size(measured):
    waist_m, inseam_m = measured
    
    size_chart = [
        (27.0, 28.0, "S Size (Small)"),
        (27.5, 30.0, "M Size (Medium)"),
        (28.0, 32.0, "L Size (Large)"),
        (29.0, 34.0, "XL Size (Extra Large)")
    ]
    
    size_ret = None
    min_dif = float("inf")
    
    def find_diff(size_tuple, measured):

        diff = sum(abs(size_tuple[i] - measured[i]) for i in range(2))
        return diff

    for size in size_chart:
        dif = find_diff(size, measured)
        if dif < min_dif:
            min_dif = dif
            size_ret = size[2]  

    return size_ret if size_ret else "No matching size"

waist_measurement = cm_to_inches(get_circumference(length_waist_lower/2))
inseam_length_measurement = cm_to_inches(inseam_len)
size = get_pants_size((waist_measurement, inseam_length_measurement))
print(f"Pant size is: {size}")







cv.imshow('final',img)
cv.waitKey(0)



result = pose.process(img)

if result.pose_landmarks:
    land_spec = mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=2)
    line_spec = mp_draw.DrawingSpec(color=(48,86,100), thickness=2)
    
    mp_draw.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS, land_spec, line_spec)
    
cv.imshow('pose', img)
cv.waitKey(0)







