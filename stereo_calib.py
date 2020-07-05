import numpy as np
import cv2

# local modules
from common import splitfn

# built-in modules
import os

h = None
w = None
pattern_size = (9, 6)


def preprocess(name):
    import sys
    import getopt
    from glob import glob

    img_mask = './' + name + '/' + name + '??.jpg'

    img_names = glob(img_mask)
    square_size = 1.0

    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    obj_points = []
    img_points = []
    
    # print(pattern_points)
    global h
    global w
    h, w = cv2.imread(img_names[0], cv2.IMREAD_GRAYSCALE).shape[:2]  # TODO: use imquery call to retrieve results

    def processImage(fn):
        # print('processing %s... ' % fn)
        img = cv2.imread(fn, 0)
        if img is None:
            print("Failed to load", fn)
            return None
        assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
        found, corners = cv2.findChessboardCorners(img, pattern_size)
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
            cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        if not found:
            print('chessboard not found')
            return None
        return (corners.reshape(-1, 2), pattern_points)

    chessboards = [processImage(fn) for fn in img_names]

    chessboards = [x for x in chessboards if x is not None]
    for (corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)

    rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
    print()
    print(name)
    print("RMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())

    return img_points, camera_matrix, dist_coefs


if __name__ == '__main__':
    pls, cml, dcl = preprocess('left')
    prs, cmr, dcr = preprocess('right')
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    obj_points = [pattern_points] * len(pls)
    print("\nstereo calibrate:")
    # cmr, roiR = cv2.getOptimalNewCameraMatrix(cmr, dcr,(w, h), 1, (w, h))
    # cml, roiR = cv2.getOptimalNewCameraMatrix(cml, dcl,(w, h), 1, (w, h))
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(obj_points, pls, prs, cml, dcl, cmr, dcr, (w, h))
    np.save('cameraMatrix1', cameraMatrix1)
    np.save('distCoeffs1', distCoeffs1)
    np.save('cameraMatrix2', cameraMatrix2)
    np.save('distCoeffs2', distCoeffs2)
    np.save('R', R)
    np.save('T', T)
    print('cameraMatrix1\n', cameraMatrix1)
    print('distCoeffs1\n', distCoeffs1)
    print('cameraMatrix2\n', cameraMatrix2)
    print('distCoeffs2\n', distCoeffs2)
    print('R\n', R)
    print('T\n', T)
    # print(help(cv2.stereoCalibrate))
    