#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility
from __future__ import print_function

import sys
import numpy as np
import cv2 as cv
from random import randint

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def main():
    print('loading images...')
    name1, name2 = sys.argv[1:]
    imgL = cv.imread(name1)  # 'match_input/left_rectified.png'
    imgR = cv.imread(name2)  # 'match_input/right_rectified.png'

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    # print(disp.max())
    # visualize
    pattern_size = (9, 6)
    point_size = 1
    point_color = (0, 0, 255) # BGR
    thickness = 4
    h, w = imgL.shape[:2]
    points = []
    for _ in range(6):
        points.append((randint(1, w), randint(1, h)))
    found, corners = cv.findChessboardCorners(imgL, pattern_size)
    points = [tuple(map(int, p[0])) for p in corners]
    
    for point in points:
        cv.circle(imgL, point, point_size, point_color, thickness)
    cv.imwrite('output/matchL.png', imgL)
    for point in points:
        try:
            p = (int(point[0] - disp[point[0], point[1]]), point[1]) # disp map
        # point[0] -= 15
            cv.circle(imgR, p, point_size, point_color, thickness)
        except:
            pass
    cv.imwrite('output/matchR.png', imgR)

    print('generating 3d point cloud...',)
    f = 0.8*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv.reprojectImageTo3D(disp, Q)
    colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply(out_fn, out_points, out_colors)
    print('%s saved' % out_fn)

    cv.imshow('left', imgL)
    cv.imshow('disparity', (disp-min_disp)/num_disp)
    cv.waitKey()

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
