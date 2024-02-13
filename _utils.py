def inside_area(face_box, area_box):
    res = 0
    fx1, fy1, fx2, fy2 = face_box
    ax1, ay1, ax2, ay2 = area_box
    if int(fx1) > ax1:
        res += 1
    if int(fy1) > ay1:
        res += 1
    if int(fx2) < ax2:
        res += 1
    if int(fy2) < ay2:
        res += 1
    return res == 4
    

def face_to_box(id, face_box, obj_box):
    res = 0
    fx1, fy1, fx2, fy2 = face_box
    ox1, oy1, ox2, oy2 = obj_box
    
    # if (fx1, fy1) is inside (ox1, oy1)
    if fx1 > ox1 and fy1 > oy1:
        res += 1
    # if (fx2, fy2) is inside (ox2, oy2)
    if fx2 < ox2 and fy2 < oy2:
        res += 1
    # if (fx1, fy2) is inside (ox1, oy2)
    if fx1 > ox1 and fy2 < oy2:
        res += 1
    # if (fx2, fy1) is inside (ox2, oy1)
    if fx2 < ox2 and fy1 > oy1:
        res += 1
    print(f"{res >= 3} belong to {id}")
    return res >= 3