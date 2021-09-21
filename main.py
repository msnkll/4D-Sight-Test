import numpy as np
import cv2 as cv

objpoints=np.load("vr3d.npy")
imgpoints=np.load("vr2d.npy")
mtx=np.array([[100,0,960],[0,100,540],[0,0,1]],dtype=float)
dist=np.array([[0,0,0,0,0]],dtype=float)

img=cv.imread("img1.png")

objpoints.astype(float)
imgpoints.astype(float)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera([objpoints], [imgpoints], gray.shape[::-1],cameraMatrix=mtx,distCoeffs=dist,
                                                  flags=cv.CALIB_USE_INTRINSIC_GUESS+cv.CALIB_FIX_PRINCIPAL_POINT+cv.CALIB_FIX_K1
                                                 +cv.CALIB_FIX_K2+cv.CALIB_FIX_K3+cv.CALIB_FIX_K4+cv.CALIB_FIX_K5+cv.CALIB_FIX_K6+
                                                  cv.CALIB_ZERO_TANGENT_DIST)

focal_length=mtx[0,0]
pp1=mtx[0,2]
pp2=mtx[1,2]

def feature_detection(path1,path2):
    MIN_MATCH_COUNT = 10

    img1 = cv.imread(path1)
    img2 = cv.imread(path2)

    
    sift = cv.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        cv.decomposeHomographyMat
        h, w, d = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))


    return src_pts,dst_pts

src_pts,dst_pts=feature_detection("img1.png","img2.png")
"""M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
ret,r1,t1,n1=cv.decomposeHomographyMat(M,mtx)"""
Essential_mtx=cv.findEssentialMat(src_pts,dst_pts,focal_length,(pp1,pp2),method=cv.RANSAC)[0]
Essential_mtx=np.array(Essential_mtx)
r11,r12,t11=cv.decomposeEssentialMat(Essential_mtx)
print("Camera pose transformation matrices in img2 w.r.t img1.. \n Rotation: \n",r11,"\n Translation: \n",t11)

src_pts,dst_pts=feature_detection("img1.png","img3.png")
Essential_mtx=cv.findEssentialMat(src_pts,dst_pts,focal_length,(pp1,pp2),method=cv.RANSAC)[0]
Essential_mtx=np.array(Essential_mtx)
r21,r22,t21=cv.decomposeEssentialMat(Essential_mtx)
print("---------------------------------------------------------")
print("Camera pose transformation matrices in img3 w.r.t img1.. \n Rotation: \n",r21,"\n Translation: \n",t21)

"""src_pts,dst_pts=feature_detection("img1.png","img3.png")
M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
ret,r2,t2,n2=cv.decomposeHomographyMat(M,mtx)"""

f = open("OUTPUT.txt", "a")
f.write("Camera pose transformation matrices in img2 w.r.t img1.. \n Rotation: \n"+str(r11)+"\n Translation: \n"+str(t11))
f.write("\n---------------------------------------------------------\n")
f.write("Camera pose transformation matrices in img2 w.r.t img1.. \n Rotation: \n"+str(r21)+"\n Translation: \n"+str(t21))
f.close()


