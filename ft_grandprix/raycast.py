import numpy as np
# from numba import njit

def rectify_shit(ranges, positions, yaws):
    pass

# @njit(cache=True)
def fakelidar(orig_x, orig_y, dt, rangefinders, cosines, sines, eps=2):
    scan = np.empty(cosines.shape)
    points = np.empty((*cosines.shape, 2))
    for i in range(rangefinders):
        x, y = orig_x, orig_y
        dx, dy = cosines[i], sines[i]
        distance = 0
        nearest = dt[int(y), int(x)]
        while nearest > eps and 0 <= x <= dt.shape[1] and 0 <= y <= dt.shape[0]:
            distance += nearest
            x += dx * nearest
            y += dy * nearest
            nearest = dt[int(y), int(x)]
        scan[i] = distance
        points[i][0] = x
        points[i][1] = y
    return scan, points

if __name__ == "__main__":
    from scipy.ndimage import distance_transform_edt
    from PIL import Image
    image = Image.open('track.png')
    dt = distance_transform_edt(image)
    rangefinders = 1440
    angles = np.linspace(0, 2*np.pi, rangefinders, endpoint=False)
    cosines = np.cos(angles)
    sines = np.sin(angles)
    scan, points = fakelidar(748, 490, dt, rangefinders, cosines, sines)
