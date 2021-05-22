#!/usr/bin/env python
# coding: utf-8

# In[52]:


import cv2
import numpy as np
import time
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'notebook')

drawing = False
ix,iy = -1,-1
color = 0
drawn_lines = [] #np.empty(shape=[0, 5])
spatial_lines = np.empty(shape=[0, 6]) #elements are two end points
origin = [0, 0]
v_mat = np.empty(shape=[4, 4])
mean_h3d = 0.
mean_h2d = 0.
catalogue = []
catalogue_index_list = []
origin = []
ex_index = 0 # 実行中のindex
axis_color = [0,0,0] #axis_color[0] -> a color of axis X
vanishing_points = []
mode = 0 #0->drawing mode, 1->reconstructing mode, 2->rotating mode

def gauss_func(x, sigma):
     return (1 / math.sqrt(2 * math.pi * (sigma ** 2))) * math.exp(-(x - 0) ** 2 / (2 * (sigma ** 2)))

def vector_cos(a, b):
    ax1, ay1, ax2, ay2, _ = a
    bx1, by1, bx2, by2, _ = b
    ax_1 = ax1 - origin[0]
    ay_1 = ay1 - origin[1]
    bx_1 = bx1 - origin[0]
    by_1 = by1 - origin[1]
    ax_2 = ax2 - origin[0]
    ay_2 = ay2 - origin[1]
    bx_2 = bx2 - origin[0]
    by_2 = by2 - origin[1]
    if (abs(ax_1) + abs(ay_1) > abs(ax_2) + abs(ay_2)):
        ax = ax_1
        ay = ay_1
    else:
        ax = ax_2
        ay = ay_2
    if (abs(bx_1) + abs(by_1) > abs(bx_2) + abs(by_2)):
        bx = bx_1
        by = by_1
    else:
        bx = bx_2
        by = by_2
    d = ax * bx + ay * by
    r = math.sqrt((ax * ax + ay * ay) * (bx * bx + by * by))
    return d / r

def camera_direction(xline, yline, zline): 
    c1 = vector_cos(xline, yline)
    c2 = vector_cos(yline, zline)
    c3 = vector_cos(zline, xline)
    c12 = c1 * c2
    c23 = c2 * c3
    c31 = c3 * c1
    dx = -math.sqrt(c31 / (c31 - c2))
    dy = -math.sqrt(c12 / (c12 - c3))
    dz = -math.sqrt(c23 / (c23 - c1))
    print("c1 = ", c1, "c2 = ", c2, "c3 = ", c3)
    
    if (c1 > 0):
        dy = -dy
    if (c3 > 0):
        dz = -dz
        
    return [dx, dy, dz]

def camera_direction2(): 
    #vec_origin = np.array(origin)
    #vec_vp0 = np.array(vanishing_points[0])
    #vec_vp1 = np.array(vanishing_points[1])
    #vec_vp2 = np.array(vanishing_points[2])
    return camera_direction(origin + vanishing_points[0] + [0],
                           origin + vanishing_points[1] + [0],
                           origin + vanishing_points[2] + [0])


def detect_intersections(l, ls): #[[2D intersection, index], ...]
    intersections = np.empty(shape=[0, 3])
    ax, ay, bx, by, _ = l
    for i in range(len(ls)):
        cx, cy, dx, dy, _ = ls[i]
        acx = cx - ax
        acy = cy - ay
        bunbo = (bx - ax)*(dy - cy) - (by - ay)*(dx - cx)
        if (bunbo != 0):
            r = ((dy - cy)*acx - (dx - cx)*acy) / bunbo
            s = ((by - ay)*acx - (bx - ax)*acy) / bunbo
            if (0 <= r and r <= 1 and 0 <= s and s <= 1):
                intersection = [(1 - r) * ax + r * bx, (1 - r) * ay + r * by, i]
                #print(intersection)
                intersections = np.append(intersections, [intersection], axis=0)
    return intersections

def focal_length():
    vpx0, vpy0 = vanishing_points[0]
    vpx1, vpy1 = vanishing_points[1]
    hx, hy = origin
    a1 = vpy0 - vpy1
    a2 = vpx0 - vpx1
    a3 = vpx0 * vpy1 + vpx1 * vpy0
    a4 = a1 * hx - a2 * hy + a3
    a5 = a1 * a1 + a2 * a2
    d2 = ((a4) ** 2) / (a5)
    x = hx + a4 * a1 / a5
    y = hy - a4 * a2 / a5
    r2 = math.sqrt((vpx0 - x) ** 2 + (vpy0 - y) ** 2) * math.sqrt((vpx1 - x) ** 2 + (vpy1 - y) ** 2)
    return math.sqrt(r2 - d2)

def view_matrix(dirc, xline, yline, origin):
    px, py, qx, qy, _ = xline
    sx, sy, tx, ty, _ = yline
    dist = 3000 #between the origin and the camera
    dx, dy, dz = dirc
    m1 = np.zeros((4, 4)) #world -> camera
    rx = math.sqrt(dy * dy + dz * dz)
    ry = math.sqrt(1 - dx * dx)
    m1[0][0:3] = [0, -dz/rx, dy/rx]
    m1[1][0:3] = [(1 - dx * dx)/ry, - dx * dy/ry, -dx * dz/ry]
    m1[2][0:3] = dirc
    m1[3][0:4] = [0,0,0,1] 
    
    #print("origin in view_matrix() = ", origin)
    
    t1 = np.array([[1, 0, 0, origin[0]], 
                   [0, 1, 0, origin[1]], 
                   [0, 0, 1, 0], 
                   [1, 1, 1, 1]]) #trans
    
    pre_yline = ((m1).dot([[0], [100], [0], [1]]).reshape([1, -1]))
    pre_xline = ((m1).dot([[100], [0], [0], [1]]).reshape([1, -1]))
    pre_yline_argument = math.atan2(pre_yline[0,1], pre_yline[0,0])
    pre_xline_argument = math.atan2(pre_xline[0,1], pre_xline[0,0])
    yline_argument = math.atan2((ty - sy), (tx - sx))
    xline_argument = math.atan2((qy - py), (qx - px))
    yphase = yline_argument - pre_yline_argument
    xphase = xline_argument - pre_xline_argument
    print("phase(x, y) = ",np.rad2deg(xphase), np.rad2deg(yphase))
    r1 = np.array([[math.cos(yphase), -math.sin(yphase), 0, 0],
                  [math.sin(yphase), math.cos(yphase), 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]) #rotate
    
    r2 = np.identity(4)
    if (abs(xphase - yphase) > 0.1 and abs(abs(xphase - yphase) - 2 * math.pi) > 0.1 
        and abs(abs(xphase - yphase) - math.pi) > 0.1):
        theta = yline_argument * 2
        r2[0][0:2] = [math.cos(theta), math.sin(theta)]
        r2[1][0:2] = [math.sin(theta), -math.cos(theta)]
    
    return t1.dot(r2).dot(r1).dot(m1)

def camera_matrix():
    return np.linalg.inv(v_mat) 

def make_x_axis_aligned_line(line, spatial_points):
    xline_ix, xline_iy, xline_x, xline_y, _ = line 
    #print("2Dline =", line)
    unit_10 = np.array([10, 0, 0, 1]) + np.concatenate([(spatial_points), np.array([0])])
    unit_20 = np.array([20, 0, 0, 1]) + np.concatenate([(spatial_points), np.array([0])])
    im_unit_10x = v_mat[0].dot(unit_10)
    im_unit_20x = v_mat[0].dot(unit_20)
    im_unit_10y = v_mat[1].dot(unit_10)
    im_unit_20y = v_mat[1].dot(unit_20)
    #print("im_unit_10:", im_unit_10x, im_unit_10y)
    #print("im_unit_20:", im_unit_20x, im_unit_20y)
    bunbo1 = (im_unit_10x - im_unit_20x)# * 2
    bunbo2 = (im_unit_10y - im_unit_20y)# * 2
    if (abs(bunbo1) > abs(bunbo2)):
        s = (xline_ix - im_unit_20x) / bunbo1
        t = (xline_x - im_unit_20x) / bunbo1
    else:
        s = (xline_iy - im_unit_20y) / bunbo2
        t = (xline_y - im_unit_20y) / bunbo2
        
    #s = (xline_ix - im_unit_20x) / bunbo1# + (xline_iy - im_unit_20y) / bunbo2
    #t = (xline_x - im_unit_20x) / bunbo1# + (xline_iy - im_unit_20y) / bunbo2
    #print("s, t = ", s, t)
    return [(unit_10 * s + unit_20 * (1 - s))[0:3], (unit_10 * t + unit_20 * (1 - t))[0:3]]

def make_y_axis_aligned_line(line, spatial_points):
    yline_ix, yline_iy, yline_x, yline_y, _ = line
    unit_10 = np.array([0, 50, 0, 1]) + np.concatenate([(spatial_points), np.array([0])])
    unit_20 = np.array([0, 100, 0, 1]) + np.concatenate([(spatial_points), np.array([0])])
    im_unit_10x = v_mat[0].dot(unit_10)
    im_unit_20x = v_mat[0].dot(unit_20)
    im_unit_10y = v_mat[1].dot(unit_10)
    im_unit_20y = v_mat[1].dot(unit_20)
    bunbo1 = (im_unit_10x - im_unit_20x)# * 2
    bunbo2 = (im_unit_10y - im_unit_20y)# * 2
    if (abs(bunbo1) > abs(bunbo2)):
        s = (yline_ix - im_unit_20x) / bunbo1
        t = (yline_x - im_unit_20x) / bunbo1
    else:
        s = (yline_iy - im_unit_20y) / bunbo2
        t = (yline_y - im_unit_20y) / bunbo2
    return [(unit_10 * s + unit_20 * (1 - s))[0:3], (unit_10 * t + unit_20 * (1 - t))[0:3]]

def make_z_axis_aligned_line(line, spatial_points):
    zline_ix, zline_iy, zline_x, zline_y, _ = line
    unit_10 = np.array([0, 0, 50, 1]) + np.concatenate([(spatial_points), np.array([0])])
    unit_20 = np.array([0, 0, 100, 1]) + np.concatenate([(spatial_points), np.array([0])])
    im_unit_10x = v_mat[0].dot(unit_10)
    im_unit_20x = v_mat[0].dot(unit_20)
    im_unit_10y = v_mat[1].dot(unit_10)
    im_unit_20y = v_mat[1].dot(unit_20)
    bunbo1 = (im_unit_10x - im_unit_20x)# * 2
    bunbo2 = (im_unit_10y - im_unit_20y)# * 2
    if (abs(bunbo1) > abs(bunbo2)):
        s = (zline_ix - im_unit_20x) / bunbo1
        t = (zline_x - im_unit_20x) / bunbo1
    else:
        s = (zline_iy - im_unit_20y) / bunbo2
        t = (zline_y - im_unit_20y) / bunbo2
    return [(unit_10 * s + unit_20 * (1 - s))[0:3], (unit_10 * t + unit_20 * (1 - t))[0:3]]

def init_3d_lines(ls, triplet):
    line0 = make_x_axis_aligned_line(ls[triplet[0]], np.array([0,0,0]))
    line1 = make_y_axis_aligned_line(ls[triplet[1]], np.array([0,0,0]))
    line2 = make_z_axis_aligned_line(ls[triplet[2]], np.array([0,0,0]))
    return [line0, line1, line2]

def length_2d(d_line):
    ix, iy, x, y, _ = d_line
    return math.sqrt((ix - x) ** 2 + (iy - y) ** 2)

def length_3d(s_line):
    #print(s_line)
    dir = s_line[0] - s_line[1]
    return math.sqrt(sum(dir ** 2))

def init_means(s_lines, d_lines):
    l3x = math.sqrt(sum((s_lines[0][0] - s_lines[0][1])**2))
    l3y = math.sqrt(sum((s_lines[1][0] - s_lines[1][1])**2))
    l3z = math.sqrt(sum((s_lines[2][0] - s_lines[2][1])**2))
    return [(l3x + l3y + l3z) / 3, (length_2d(d_lines[0]) + length_2d(d_lines[1]) + length_2d(d_lines[2])) / 3]

def find_spatial_point(sp_line, dr_line, point):
    px, py = point
    dix, diy, dx, dy, _ = dr_line
    bunbox = dix - dx
    bunboy = diy - dy
    if (bunbox != 0 and bunboy != 0):
        t = ((px - dx) / bunbox + (py - dy) / bunboy) / 2
    elif (bunbox != 0):
        t = (px - dx) / bunbox
    else:
        t = (py - dy) / bunboy
    s_iv, s_v = sp_line
    return (s_iv * t + s_v * (1 - t))[0:3]

def make_through_two_points(point1, point2, line):#3D交点と2D線分から、3D線分を計算
    ix, iy, x, y, _ = line
    #print(point1)
    point_vector1 = np.concatenate([point1, np.array([1])])
    point_vector2 = np.concatenate([point2, np.array([1])])
    drawn_intersection1x = v_mat[0].dot(point_vector1)
    drawn_intersection2x = v_mat[0].dot(point_vector2)
    drawn_intersection1y = v_mat[1].dot(point_vector1)
    drawn_intersection2y = v_mat[1].dot(point_vector2)
    bunbox = drawn_intersection1x - drawn_intersection2x
    bunboy = drawn_intersection1y - drawn_intersection2y
    if (abs(bunbox) > abs(bunboy)):
        s = ((ix - drawn_intersection2x)/bunbox)
        t = ((x - drawn_intersection2x) / bunbox)
    else:
        s = (iy - drawn_intersection2y)/bunboy
        t = (y - drawn_intersection2y) / bunboy
    return [point1 * s + point2 * (1 - s), point1 * t + point2 * (1 - t)]

def q_coverage(line, intersection1, intersection2):
    ix, iy, x, y, _ = line
    ox1, oy1, _ = intersection1
    ox2, oy2, _ = intersection2
    segment = abs(ox1 - ox2) + abs(oy1 - oy2)
    length = abs(x - ix) + abs(y - iy)
    return segment / length

def q_coverage2(ox, oy, line):
    ix, iy, x, y, _ = line
    length = abs(x - ix) + abs(y - iy)
    dist = max(abs(ox - ix) + abs(oy - iy), abs(ox - x) + abs(oy - y))
    return dist / length

def q_axis(unit, candidate_line):
    sigma = 0.01135
    c1, c2 = candidate_line
    direction = c1 - c2
    #print("candidate_line", candidate_line, "direction = ", direction)
    length = np.sqrt(sum(direction**2))
    #print (abs(unit.dot(direction)) / length, gauss_func(0, sigma))
    #print("gauss_func = ", gauss_func(0, sigma), "length = ", length)
    if (length == 0):
        return 0
    return (gauss_func(1 - (abs(unit.dot(direction)) / length), sigma)) / gauss_func(0, sigma)

def q_ortho(candidate_line, ad_line):
    sigma = 0.086
    c1, c2 = candidate_line
    c_direction = c1 - c2
    c_length = np.sqrt(sum(c_direction**2))
    a1, a2 = ad_line
    a_direction = a1 - a2
    a_length = np.sqrt(sum(a_direction**2))
    #print(c_direction, a_direction)
    #print(abs(c_direction.dot(a_direction)) / (c_length * a_length))
    return gauss_func(abs(c_direction.dot(a_direction)) / (c_length * a_length), sigma) / gauss_func(0, sigma)

def q_planar(candidate_line, ad_line1, ad_line2):
    sigma = 0.086
    c1, c2 = candidate_line
    c_direction = c1 - c2
    c_length = np.sqrt(sum(c_direction**2))
    c_direction = c_direction / c_length
    a1, a2 = ad_line1
    a_direction = a1 - a2
    b1, b2 = ad_line2
    b_direction = b1 - b2
    n = np.array([a_direction[1] * b_direction[2] - a_direction[2] * b_direction[1],
                 a_direction[2] * b_direction[0] - a_direction[0] * b_direction[2],
                 a_direction[0] * b_direction[1] - a_direction[1] * b_direction[0]])
    n_length = np.sqrt(sum(n**2))
    if (n_length == 0):
        n = 100 * n
    else:
        n = n / n_length
    #print(abs(c_direction.dot(n)))
    return gauss_func(abs(c_direction.dot(n)), sigma) / gauss_func(0, sigma)

def q_tangent(candidate_line, ad_line):
    sigma = 0.086
    c1, c2 = candidate_line
    c_direction = c1 - c2
    c_length = np.sqrt(sum(c_direction**2))
    a1, a2 = ad_line
    a_direction = a1 - a2
    a_length = np.sqrt(sum(a_direction**2))
    #print(c_direction, a_direction)
    #print(abs(c_direction.dot(a_direction)) / (c_length * a_length))
    return gauss_func(1 - abs(c_direction.dot(a_direction)) / (c_length * a_length), sigma) / gauss_func(0, sigma)


def draw_true_intersections(intersection_id, intersections):
    for i in range(len(intersection_id)):
        px, py, _ = intersections[intersection_id[i]]
        cv2.circle(img,(int(px), int(py)), 3, (0, 1, 1), -1)
        
def distance_point_line(point, line):
    a, b, c = line[0]
    d, e, f = line[1]
    p, q, r = point
    da = d - a
    eb = e - b
    fc = f - c
    pa = p - a
    qb = q - b
    rc = r - c
    t = (da * pa + eb * qb + fc * rc) / (da * da + eb * eb + fc * fc)
    if (t > 1.1):
        return 1000
    else:
        n = np.array([da * t - pa, eb * t - qb, fc * t - rc])
        return math.sqrt(n.dot(n))

def detect_line_near_point(sp_lines, point):
    index_list = []
    for i in range(len(sp_lines)):
        line = sp_lines[i]
        d = distance_point_line(point, line)
        if (d < 10):
            index_list = index_list + [i]
    return index_list

def make_adjacent_list(spatial_point, ref_list):#ある3D点の近傍を通る3D直線のリスト
    global spatial_lines, catalogue, catalogue_index_list
    r = []
    line_number = len(spatial_lines)
    ref_number = len(ref_list)
    for i in range(line_number):
        catalogue_index = find_catalogue_index(i)
        if (catalogue_index < 0):
            line = spatial_lines[i]
            d = distance_point_line(spatial_point, line)
            if (d < 10):
                r = r + [line]
    
    for i in range(ref_number):
        line_index, candidate_index = ref_list[i]
        catalogue_index = find_catalogue_index(line_index)
        if (catalogue_index < 0):
            #print(ref_list[i], catalogue_index_list)
            continue
        line = catalogue[catalogue_index][candidate_index][0]
        d = distance_point_line(spatial_point, line)
        if (d < 10):
            r = r + [line]
    
    return r

def accept(max_value, second_max_value):
    t_scorehigh = 0.98
    t_score = 0.75
    t_ambiguity = 0.8
    return (max_value > t_scorehigh) or ((max_value > t_score) and second_max_value / max_value < t_ambiguity)

def find_catalogue_index(p_id):
    try:
        i = catalogue_index_list.index(p_id)
    except:
        return -1
    else:
        return i

def double_sort(d1, d2):#任意の[a, b],[a, c]で、b = cなら統合、b != cなら[]を返す
    i = 0
    j = 0
    length_d1 = len(d1)
    length_d2 = len(d2)
    r = []
    
    while(i < length_d1 and j < length_d2):
        a, b = d1[i]
        c, d = d2[j]
        if (a > c):
            r = r + [[c, d]]
            j = j + 1
        elif (a < c):
            r = r + [[a, b]]
            i = i + 1
        elif (a == c and b == d):
            r = r + [[a, b]]
            i = i + 1
            j = j + 1
        else:
            return [-1, -1]
    
    if (i < length_d1):
        return r + d1[i:length_d1]
    elif (j < length_d2):
        return r + d2[j:length_d2]
    else:
        return r

def apply_delayed_lines(pages):
    global catalogue, catalogue_index_list, spatial_lines
    i = 0
    if (pages == []):
        return
    else:
        i = len(pages) - 1
        while(i >= 0):
            line_index, candidate_index =  pages[i]
            catalogue_index = find_catalogue_index(line_index)
            if (catalogue_index < 0):
                print("no longer paged")
                i = i - 1
                continue
            candidate = catalogue[catalogue_index][candidate_index]
            spatial_lines[int(line_index)] = candidate[0]
            i = i - 1
            catalogue_index_list.pop(catalogue_index)
            catalogue.pop(catalogue_index)
        return
    
    
def orthocenter(points):
    xa, ya = points[0]
    xb, yb = points[1]
    xc, yc = points[2]
    
    arr1 = np.array([[-xb * xc - ya * ya, ya, 1], 
                     [-xa * xc - yb * yb, yb, 1], 
                     [-xa * xb - yc * yc, yc, 1]])
    arr2 = np.array([[xa, -xa * xa - yb * yc, 1], 
                     [xb, -xb * xb - ya * yc, 1], 
                     [xc, -xc * xc - ya * yb, 1]])
    arr3 = np.array([[xa, ya, 1], 
                     [xb, yb, 1], 
                     [xc, yc, 1]])
    bunbo = np.linalg.det(arr3)
    
    return [np.linalg.det(arr1) / bunbo, np.linalg.det(arr2) / bunbo]
        

    

def draw_line(event,x,y,flags,param): #lineを引かれる度に計算
    global mode, ix,iy,drawing, color, drawn_lines, spatial_lines, origin, v_mat, mean_h3d, mean_h2d, catalogue, catalogue_index_list, origin
    if (mode == 0 or mode == 1):
        img[:] = 0
        #nn_us_0 = "{}{:.0f}{}".format('coffee :', a, 'yen')
        if (mode == 0):
            cv2.putText(img, "Drag lines! When you finish drawing a sketch, press 'ESC' key!", (40, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, "If you want to delete a dragged line, press 'z' key!", (40, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "Wait...", (40, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1, cv2.LINE_AA)
        for i in range(50):
            for j in range(50):
                cv2.circle(img,(40 * i, 40 * j), 1, (1,1,1), -1)
        for i in range(len(drawn_lines)):
            ax, ay, bx, by, col = drawn_lines[i]
            if (col == 1):
                cv2.line(img,(ax,ay),(bx,by),(0,0,1))
            elif (col == 2):
                cv2.line(img,(ax,ay),(bx,by),(0,1,0))
            elif (col == 3):
                cv2.line(img,(ax,ay),(bx,by),(1,0,0))
            else:
                cv2.line(img,(ax,ay),(bx,by),(1,1,1))

        if event == cv2.EVENT_LBUTTONDOWN:
            ix,iy = x,y
            #cv2.circle(img,(ix, iy), 3, (1, 1, 1), -1)
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                cv2.line(img,(ix,iy),(x,y),(1,1,1))
                #img[:] = 0
        elif (event == cv2.EVENT_LBUTTONUP):
            drawing = False
            cv2.line(img,(ix,iy),(x,y),(1,1,1))       
            new_line = [ix, iy, x, y, color]
            intersections = detect_intersections(new_line, drawn_lines)
            drawn_lines = drawn_lines + [new_line] #np.append(drawn_lines, [new_line], axis=0)
            
            
            
def reconstruct(new_line): #2D -> 3D
    global drawing, spatial_lines, origin, v_mat, mean_h3d, mean_h2d, catalogue, catalogue_index_list, origin, drawn_lines, ex_index, origin
    
    color = new_line[4]
    intersections = detect_intersections(new_line, drawn_lines[0:ex_index])
    print("intersections :", intersections)
    for foo in range(1):
            
        if (len(intersections) == 0 or (len(intersections) == 1 and color == 0)):
            drawn_lines.pop(ex_index)
            if (ex_index + 5 < len(drawn_lines)):
                drawn_lines.insert(ex_index + 5, new_line)
            else:
                drawn_lines = drawn_lines + [new_line]
        elif (len(intersections) == 1 and color != 0):
            ox, oy, p_index = intersections[0]
            spartial_partner_line = spatial_lines[int(p_index)]
            drawn_partner_line = drawn_lines[int(p_index)]
            
            spatial_point = find_spatial_point(spartial_partner_line, drawn_partner_line, [ox, oy])
            #print("spatial_point: ",spatial_point)
            if (color == axis_color[0]):
                spatial_lines = np.append(spatial_lines, [make_x_axis_aligned_line(new_line, spatial_point)], axis=0)
            elif (color == axis_color[1]):
                spatial_lines = np.append(spatial_lines, [make_y_axis_aligned_line(new_line, spatial_point)], axis=0)
            elif (color == axis_color[2]):
                spatial_lines = np.append(spatial_lines, [make_z_axis_aligned_line(new_line, spatial_point)], axis=0)
                
            draw_true_intersections([0], intersections)
            ex_index += 1
            #print("spatial_lines:",spatial_lines)
            
        elif (ex_index >= 3):
            candidate_lines = [] # np.empty(shape = [0,2,3])
            intersection_id = [] #3Dintersectionに対応する2Dintersection
            max_value = 0
            new_mean = 0.0
            best_candidate_line = [] #[[[ix, iy, iz],[x, y, z]], max_value, [partner_id, candidate_index], means_max_values]
            second_max_value = 0
            second_best_candidate_line = []
            q_a = 0
            q_p = 0
            q_o = 0
            q_t = 0
            q = 0
            
            #####additional_candidate_line#####
            if (color > 0):
                for i in range(len(intersections)):
                    ox, oy, p_index = intersections[i]
                    catalogue_index = find_catalogue_index(p_index)
                    drawn_partner_line = drawn_lines[int(p_index)]
                    if (catalogue_index < 0):
                        j = 0
                    else :
                        j = len(catalogue[catalogue_index]) - 1
                    while (j >= 0):
                        if (catalogue_index < 0):
                            spartial_partner_line = spatial_lines[int(p_index)]
                            reference_list = [] 
                            mean_max_value = 0.0
                        else:
                            spartial_partner_line = catalogue[catalogue_index][j][0]
                            reference_list = catalogue[catalogue_index][j][2] + [[p_index,j]]
                            mean_max_value = catalogue[catalogue_index][j][3] * (len(reference_list))

                        spatial_point = find_spatial_point(spartial_partner_line, drawn_partner_line, [ox, oy])
                        #print("spatial_point: ",spatial_point)
                        if (color == axis_color[0]):
                            candidate_line = make_x_axis_aligned_line(new_line, spatial_point)
                            q_a = q_axis(np.array([1,0,0]), candidate_line)
                        elif (color == axis_color[1]):
                            candidate_line = make_y_axis_aligned_line(new_line, spatial_point)
                            q_a = q_axis(np.array([0,1,0]), candidate_line)
                        elif (color == axis_color[2]):
                            candidate_line = make_z_axis_aligned_line(new_line, spatial_point)
                            q_a = q_axis(np.array([0,0,1]), candidate_line)
                        else:
                            print("fatal error. color =", color)
                        q_c = q_coverage2(ox, oy, new_line)
                        q = 0.4 * q_a + 0.4 * q_c + 0.2 * q_a * q_c
                        #print(q_c)
                        new_mean = (mean_max_value + q) / (1 + len(reference_list))
                        candidate_lines = candidate_lines +  [[candidate_line, q, reference_list, new_mean]]
                        if (q > max_value):
                            max_value = q
                            intersection_id = [i]
                            best_candidate_line = [candidate_line, q, reference_list, new_mean]
                        j = j - 1
            ############
            
            
                    
            for i in range(0, len(intersections) - 1):
                ox1, oy1, p_index1 = intersections[i]
                drawn_partner_line1 = drawn_lines[int(p_index1)]
                catalogue_index1 = find_catalogue_index(p_index1)
                if (catalogue_index1 < 0):
                    ii = 0
                else:
                    ii = len(catalogue[catalogue_index1]) - 1
                    
                while (ii >= 0):
                    if (catalogue_index1 < 0):
                        spartial_partner_line1 = spatial_lines[int(p_index1)]
                        reference_list1 = []
                        mean_max_value1 = 0.0
                    else:
                        spartial_partner_line1 = catalogue[catalogue_index1][ii][0]
                        reference_list1 = catalogue[catalogue_index1][ii][2] + [[p_index1, ii]]
                        mean_max_value1 = catalogue[catalogue_index1][ii][3] * len(reference_list1)
                    spatial_point1 = find_spatial_point(spartial_partner_line1, drawn_partner_line1, [ox1, oy1])
                    
                    for j in range(i + 1, len(intersections)):
                        ox2, oy2, p_index2 = intersections[j]
                        #spartial_partner_line2 = spatial_lines[int(p_index2)]
                        drawn_partner_line2 = drawn_lines[int(p_index2)]
                        catalogue_index2 = find_catalogue_index(p_index2)
                        if (catalogue_index2 < 0):
                            jj = 0
                        else:
                            jj = len(catalogue[catalogue_index2]) - 1
                            
                        while (jj >= 0):
                            if (catalogue_index2 < 0):
                                spartial_partner_line2 = spatial_lines[int(p_index2)]
                                reference_list = reference_list1 
                                mean_max_value2 = 0.0
                            else:
                                spartial_partner_line2 = catalogue[catalogue_index2][jj][0]
                                reference_list2 = catalogue[catalogue_index2][jj][2] + [[p_index2, jj]]
                                mean_max_value2 = catalogue[catalogue_index2][jj][3] * len(reference_list2)
                                reference_list = double_sort(reference_list1, reference_list2)
                                
                            if (reference_list == [-1, -1]):
                                jj = jj - 1
                                continue
                            #print(reference_list)
                            spatial_point2 = find_spatial_point(spartial_partner_line2, drawn_partner_line2, [ox2, oy2])
                            candidate_line = make_through_two_points(spatial_point1, spatial_point2, new_line)
                            if (length_3d(candidate_line) / mean_h3d > length_2d(new_line) * 3 / mean_h2d 
                                and len(intersections) > 2 and (not(max_value == 0 and i+j == len(intersections) * 2 - 3))):
                                print("skip")
                                jj = jj - 1
                                #print(i, j, ii, jj)
                                continue
                            q_c = q_coverage(new_line, intersections[i], intersections[j])
                            q_a = 0
                            q_p = 0
                            q_o = 0
                            q_t = 0
                            if (color == 1):
                                q_a = q_axis(np.array([1,0,0]), candidate_line)
                            elif (color == 2):
                                q_a = q_axis(np.array([0,1,0]), candidate_line)
                            elif (color == 3):
                                q_a = q_axis(np.array([0,0,1]), candidate_line)
                            else:
                                adjacent_list1 = make_adjacent_list(spatial_point1, reference_list)
                                adjacent_list2 = make_adjacent_list(spatial_point2, reference_list)
                                #print(index_list1)
                                for i0 in range(len(adjacent_list1)):
                                    q_o = max(q_ortho(candidate_line, adjacent_list1[i0]), q_o)
                                    q_t = max(q_tangent(candidate_line, adjacent_list1[i0]), q_t)
                                for i1 in range(len(adjacent_list2)):
                                    q_o = max(q_ortho(candidate_line, adjacent_list2[i1]), q_o)
                                    q_t = max(q_tangent(candidate_line, adjacent_list2[i1]), q_t)

                                if (len(adjacent_list1) >= 2):
                                    for i2 in range(0, len(adjacent_list1) - 1):
                                        for j2 in range(i2 + 1, len(adjacent_list1)):
                                            q_p = max(q_planar(candidate_line, adjacent_list1[i2], adjacent_list1[j2]), q_p)
                                if (len(adjacent_list2) >= 2):
                                    for i2 in range(0, len(adjacent_list2) - 1):
                                        for j2 in range(i2 + 1, len(adjacent_list2)):
                                            q_p = max(q_planar(candidate_line, adjacent_list2[i2], adjacent_list2[j2]), q_p)

                            q_g = max(q_a, q_o, q_p, q_t)
                            q = 0.4 * q_c + 0.4 * q_g + 0.2 * q_c * q_g
                            new_mean = (mean_max_value1 + mean_max_value2 + q) / (len(reference_list) + 1)
                            if(q > 0.4 or len(candidate_lines) == 0): #閾値を設ける
                                candidate_lines = candidate_lines +  [[candidate_line, q, reference_list, new_mean]]
                            else:
                                #print("q is too small")
                                jj = jj - 1
                                continue
                            if (q > max_value):
                                second_max_value = max_value
                                max_value = q
                                intersection_id = [i,j]
                                second_best_candidate_line = best_candidate_line
                                best_candidate_line = [candidate_line, q, reference_list, new_mean]
                            jj = jj - 1
                    ii = ii - 1
            print("best_candidate_line:", best_candidate_line)
            if (best_candidate_line == []):
                print("candidate_lines:", candidate_lines)
            print(len(best_candidate_line[2]))
            if (accept(max_value, second_max_value) or (len(best_candidate_line[2]) > 0 and new_mean > 0.90)):
                #or (len(intersections) == 2 and color == 0)):
                print("accept")
                spatial_lines = np.append(spatial_lines, [best_candidate_line[0]], axis=0)
                mean_h3d = (ex_index * mean_h3d + length_3d(best_candidate_line[0])) / (ex_index + 1)
                mean_h2d = (ex_index * mean_h2d + length_2d(new_line)) / (ex_index + 1)
                apply_delayed_lines(best_candidate_line[2])
                draw_true_intersections(intersection_id, intersections)
            else:
                print("not accept")
                spatial_lines = np.append(spatial_lines, [best_candidate_line[0]], axis=0)
                catalogue = catalogue + [candidate_lines]
                catalogue_index_list = catalogue_index_list + [ex_index]
                print(len(candidate_lines))
            print(best_candidate_line)
            print(catalogue_index_list)
            #draw_true_intersections(intersection_id, intersections)
            ex_index += 1


            
def q_vanising_point(line, intersection):
    sigma = 5
    
    ix, iy, x, y, _ = line
    px, py = intersection
    l = math.sqrt((x - ix) ** 2 + (y - iy) ** 2)
    
    if (px == np.inf):
        deg1 = py
        deg2 = np.degrees(np.arctan2(y - iy, x - ix))
        deg12 = abs(deg1 - deg2)
        if (deg12 < 180):
            deg = deg12
        else:
            deg = 360 - deg12
    else:
        mx = (ix + x) / 2
        my = (iy + y) / 2
        inner = (px - mx) * (x - ix) + (py - my) * (y - iy)
        p = math.sqrt((px - mx) ** 2 + (py - my) ** 2)
        cos_deg = abs(inner / l / p)
        if (cos_deg > 1 and cos_deg < 1.001):
            cos_deg = 1
        deg = np.degrees(np.arccos(cos_deg))
    #print("deg = ", deg, "line = ", line, "gauss = ", math.exp(- deg / (2 * sigma ** 2)))
    return l * math.exp(- deg / (2 * sigma ** 2))


def find_vanishing_point(lines, col): #vanishing_pointと、non_labelのリストを返す
    global drawn_lines
    
    member_threshold = 0.90
    
    len_lines = len(lines)
    intersections = []
    remaining = []
    for i in range(len_lines):
        ax, ay, bx, by, _ = lines[i]
        for j in range(len_lines):
            cx, cy, dx, dy, _ = lines[j]
            acx = cx - ax
            acy = cy - ay
            bunbo = (bx - ax)*(dy - cy) - (by - ay)*(dx - cx)
            if (bunbo == 0):
                deg = np.degrees(np.arctan2(by - ay, bx - ax))
                intersections += [[np.inf, deg]]
            else:
                r = ((dy - cy)*acx - (dx - cx)*acy) / bunbo
                s = ((by - ay)*acx - (bx - ax)*acy) / bunbo
                if ((0 > r or r > 1) and (0 > s or s > 1)):
                    intersections += [[(1 - r) * ax + r * bx, (1 - r) * ay + r * by]]
    
    #print(intersections)
    len_intersections = len(intersections)
    max_vote = 0
    max_index = 0
    for i in range(len_intersections):
        vote = 0
        for j in range(len_lines):
            vote += q_vanising_point(lines[j], intersections[i])
        if (max_vote < vote):
            max_vote = vote
            max_index = i
            
    vanishing_point = intersections[max_index]
    
    #labeling
    len_drawn_lines = len(drawn_lines)
    for i in range(len_drawn_lines):
        if (drawn_lines[i][4] == 0):
            vote = q_vanising_point(drawn_lines[i], vanishing_point)
            l = length_2d(drawn_lines[i])
            #print("line: ", i, "line_length = ", l, " score : ", vote / l)
            if (vote / l > member_threshold):
                drawn_lines[i][4] = col
                ix, iy, x, y, _ = drawn_lines[i]
                if (col == 1):
                    cv2.line(img,(ix,iy),(x,y),(0,0,1))
                elif (col == 2):
                    cv2.line(img,(ix,iy),(x,y),(0,1,0))
                else:
                    cv2.line(img,(ix,iy),(x,y),(1,0,0))
                
            else:
                remaining += [drawn_lines[i]]
    print("vanishing_point: ", vanishing_point, "remaining: ", remaining)
    return vanishing_point, remaining

def detect_2Dline_near_point(ox, oy, col0, col1): #return index
    index = -1
    for i in range(len(drawn_lines)):
        ix, iy, x, y, col = drawn_lines[i]
        if (col == 0 or col == col0 or col == col1):
            continue
        a1 = ix - x
        a2 = iy - y
        a3 = ix * y - iy * x
        a4 = a1 * oy - a2 * ox - a3
        distance = abs(a4)  / math.sqrt(a1 * a1 + a2 * a2)
        #print("ox, oy, ix, iy, x, y , a1, a2, a3, a4= ", ox, oy, ix, iy, x, y, a1, a2, a3, a4)
        #print("distance = ", distance)
        if (distance < 5):
            index = i
            break
    return index

def find_3d_intersection():
    global drawn_lines, spatial_lines, origin, axis_color
    list = [-1,-1,-1]
    flag = False
    for i in range(len(drawn_lines) - 1):
        color0 = drawn_lines[i][4]
        if (color0 == 0):
            continue
        intersections = detect_intersections(drawn_lines[i], drawn_lines[i + 1:])
        for j in range(len(intersections)):
            ox, oy, p_index = intersections[j] #this p_index is drawn_lines's index - (i + 1)
            color1 = drawn_lines[int(p_index) + i + 1][4]
            if (color1 == 0 or color1 == color0):
                continue
            k = detect_2Dline_near_point(ox, oy, color0, color1)
            if (k >= 0):
                list = [i, int(p_index) + i + 1, k]
                origin = [ox, oy]
                axis_color = [color0, color1, 6 - color0 - color1]
                flag = True
                break
        if (flag):
            break
    return list

def calibrate(triplet):
    global drawn_lines, spatial_lines, v_mat, mean_h3d, mean_h2d
    d0 = drawn_lines[triplet[0]]
    d1 = drawn_lines[triplet[1]]
    d2 = drawn_lines[triplet[2]]
    cam_dirc = camera_direction(d0, d1, d2) #the direction of the camera
    print("cam_dirc =", cam_dirc)
    print("d0 + d1 + d2 = ", cam_dirc[0] + cam_dirc[1] + cam_dirc[2])
    v_mat = view_matrix(cam_dirc, d0, d1, origin)
    spatial_lines = init_3d_lines(drawn_lines, triplet)
    drawn_lines.remove(d0)
    drawn_lines.remove(d1)
    drawn_lines.remove(d2)
    drawn_lines = [d0, d1, d2] + drawn_lines
    mean_h3d, mean_h2d = init_means(spatial_lines, drawn_lines)
    
    print("spatial_lines[0] = ", spatial_lines[0])
    
    ix = v_mat[0].dot(np.concatenate([(spatial_lines[0][0]), np.array([1])]))
    x = v_mat[0].dot(np.concatenate([(spatial_lines[0][1]), np.array([1])]))
    iy = v_mat[1].dot(np.concatenate([(spatial_lines[0][0]), np.array([1])]))
    y = v_mat[1].dot(np.concatenate([(spatial_lines[0][1]), np.array([1])]))
    print("line0 = ", ix, iy, x, y)
    cv2.line(img,(int(ix),int(iy)),(int(x),int(y)),(1,1,0))
    ix = v_mat[0].dot(np.concatenate([(spatial_lines[1][0]), np.array([1])]))
    x = v_mat[0].dot(np.concatenate([(spatial_lines[1][1]), np.array([1])]))
    iy = v_mat[1].dot(np.concatenate([(spatial_lines[1][0]), np.array([1])]))
    y = v_mat[1].dot(np.concatenate([(spatial_lines[1][1]), np.array([1])]))
    print("line1 = ", ix, iy, x, y)
    cv2.line(img,(int(ix),int(iy)),(int(x),int(y)),(0,1,1))
    ix = v_mat[0].dot(np.concatenate([(spatial_lines[2][0]), np.array([1])]))
    x = v_mat[0].dot(np.concatenate([(spatial_lines[2][1]), np.array([1])]))
    iy = v_mat[1].dot(np.concatenate([(spatial_lines[2][0]), np.array([1])]))
    y = v_mat[1].dot(np.concatenate([(spatial_lines[2][1]), np.array([1])]))
    print("line2 = ", ix, iy, x, y)
    cv2.line(img,(int(ix),int(iy)),(int(x),int(y)),(1,0,1))
    
def rotation_x(x):
    r = np.identity(4)
    r[1][1:3] = [math.cos(x), -math.sin(x)]
    r[2][1:3] = [math.sin(x), math.cos(x)]
    return r

def rotation_y(x):
    r = np.identity(4)
    r[0][0:3] = [math.cos(x), 0, math.sin(x)]
    r[2][0:3] = [-math.sin(x), 0, math.cos(x)]
    return r

def rotation_z(x):
    r = np.identity(4)
    r[0][0:2] = [math.cos(x), -math.sin(x)]
    r[1][0:2] = [math.sin(x), math.cos(x)]
    return r
            
def rotate(val):
    if (mode == 2):
        a = cv2.getTrackbarPos('X','my_drawing')
        b = cv2.getTrackbarPos('Y','my_drawing')
        c = cv2.getTrackbarPos('Z','my_drawing')
        #np.deg2rad()
        r_x = rotation_x(np.deg2rad(a))
        r_y = rotation_y(np.deg2rad(b))
        r_z = rotation_z(np.deg2rad(c))
        mat = v_mat.dot(r_z).dot(r_y).dot(r_x)
        img[:] = 0
        #cv2.putText(img, "Scroll three tracking bar above! Then, your sketch will 'rotate'!", (40, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "If you want to end execution, press 'ESC' key!", (40, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1, cv2.LINE_AA)
        for i in range(len(spatial_lines)):
            ix = mat[0].dot(np.concatenate([(spatial_lines[i][0]), np.array([1])]))
            iy = mat[1].dot(np.concatenate([(spatial_lines[i][0]), np.array([1])]))
            x = mat[0].dot(np.concatenate([(spatial_lines[i][1]), np.array([1])]))
            y = mat[1].dot(np.concatenate([(spatial_lines[i][1]), np.array([1])]))
            cv2.line(img,(int(ix),int(iy)),(int(x),int(y)),(1,1,1))
        #print ([r, g, b])
    
img = np.zeros((800,1000,3))

cv2.namedWindow(winname='my_drawing') # , cv2.WINDOW_NORMAL
#cv2.namedWindow(winname='my_drawing2')
cv2.setMouseCallback('my_drawing',draw_line)

cv2.createTrackbar('X','my_drawing',0,359,rotate)
cv2.createTrackbar('Y','my_drawing',0,359,rotate)
cv2.createTrackbar('Z','my_drawing',0,359,rotate)


while True: 
    cv2.imshow('my_drawing',img)
    key = cv2.waitKey(10) & 0xFF
    if (key == 27):
        mode = 1
        break
    elif (key == ord('z')):
        ix, iy, x, y, _ = drawn_lines[len(drawn_lines) - 1]
        cv2.line(img,(int(ix),int(iy)),(int(x),int(y)),(0,0,0))
        drawn_lines.pop()

arg_fvp = drawn_lines
print(drawn_lines)

for i in range(1, 4):
    vp, rem = find_vanishing_point(arg_fvp, i)
    vanishing_points += [vp]
    arg_fvp = rem

print("colored_drawn_lines;", drawn_lines)
print("vanishing_points = ",vanishing_points)

axis_color = [drawn_lines[0][4], drawn_lines[1][4], drawn_lines[2][4]]
print("axis_color = ", axis_color)

triplet = find_3d_intersection()
print("triplet = ", triplet)
calibrate(triplet)
print("calibrated_drawn_lines;", drawn_lines)

ex_index = 3
while (ex_index < len(drawn_lines)):
    #print("dl:", drawn_lines)
    reconstruct(drawn_lines[ex_index])
mode = 2
cv2.putText(img, "Scroll three tracking bar above! Then, your sketch will 'rotate'!", (40, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1, cv2.LINE_AA)
while True:
    cv2.imshow('my_drawing',img)
    #img[:] = [0, 0,0]
    if (cv2.waitKey(1) & 0xFF == 27):
        break
        
cv2.destroyAllWindows()

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection' : '3d'})
xmax = 0
xmin = 0
ymax = 0
ymin = 0
zmax = 0
zmin = 0

for i in range(len(spatial_lines)):
    x0 = spatial_lines[i][0][0]
    x1 = spatial_lines[i][1][0]
    y0 = spatial_lines[i][0][1]
    y1 = spatial_lines[i][1][1]
    z0 = spatial_lines[i][0][2]
    z1 = spatial_lines[i][1][2]
    X = [x0, x1]
    Y = [y0, y1]
    Z = [z0, z1]
    xmax = max(xmax, x0, x1)
    ymax = max(ymax, y0, y1)
    zmax = max(zmax, z0, z1)
    xmin = min(xmin, x0, x1)
    ymin = min(ymin, y0, y1)
    zmin = min(zmin, z0, z1)
    if (find_catalogue_index(i) < 0):
        ax.plot(X, Y, Z, ".-", color="#00aa00", ms=4, mew=0.5)
    else:
        ax.plot(X, Y, Z, ".-", color="#0000aa", ms=4, mew=0.5)
    
max_range = np.array([xmax - xmin, ymax - ymin, zmax - zmin]).max() * 0.5

mid_x = (xmax + xmin) * 0.5
mid_y = (ymax + ymin) * 0.5
mid_z = (zmax + zmin) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
ax.set_xlabel("X", fontsize=20)
ax.set_ylabel("Y", fontsize=20)
ax.set_zlabel("Z", fontsize=20)
    
plt.show()


# In[ ]:




