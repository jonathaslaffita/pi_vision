#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
 
 using namespace cv;
 using namespace std;
// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{6,9}; 
int i;
int main()
{
  // Creating vector to store vectors of 3D points for each checkerboard image
  vector<vector<Point3f> > objpoints;
 
  // Creating vector to store vectors of 2D points for each checkerboard image
  vector<vector<Point2f> > imgpoints;
 
  // Defining the world coordinates for 3D points
  vector<Point3f> objp;
  for(int i{0}; i<CHECKERBOARD[1]; i++)
  {
    for(int j{0}; j<CHECKERBOARD[0]; j++)
      objp.push_back(Point3f(j,i,0));
  }
 
 
  // // Extracting path of individual image stored in a given directory
  // vector<String> images;
  // // Path of the folder containing checkerboard images
  // string path = "./images/*.jpg";
 
  // glob(path, images);
 
  Mat frame, gray;
  // vector to store the pixel coordinates of detected checker board corners 
  vector<Point2f> corner_pts;
  bool success;
VideoCapture inputVideo;
inputVideo.open(0);
while (inputVideo.grab()) {
  Mat image, imageCopy;
          inputVideo.retrieve(image);
          image.copyTo(frame);
    
    cvtColor(frame,gray,COLOR_BGR2GRAY);
 
    // Finding checker board corners
    // If desired number of corners are found in the image then success = true  
    success = findChessboardCorners(gray, (Size(CHECKERBOARD[0], CHECKERBOARD[1])), CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
        + CALIB_CB_FAST_CHECK);
     
    /* 
     * If desired number of corner are detected,
     * we refine the pixel coordinates and display 
     * them on the images of checker board
    */
    if(success)
    {  
      i ++;
      TermCriteria criteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.001);
       
      // refining pixel coordinates for given 2d points.
      cornerSubPix(gray,corner_pts,Size(11,11), Size(-1,-1),criteria);
       
      // Displaying the detected corner points on the checker board
      drawChessboardCorners(frame, Size(6,9), corner_pts, success);
       
      objpoints.push_back(objp);
      imgpoints.push_back(corner_pts);
    }
 
    imshow("Image",frame);
    waitKey(0);
    if(i>10){
        inputVideo.(0);
    }
  }

  destroyAllWindows();
 
  Mat cameraMatrix,distCoeffs,R,T;
 
  /*
   * Performing camera calibration by 
   * passing the value of known 3D points (objpoints)
   * and corresponding pixel coordinates of the 
   * detected corners (imgpoints)
  */
  calibrateCamera(objpoints, imgpoints, Size(gray.rows,gray.cols), cameraMatrix, distCoeffs, R, T);
 
  cout << "cameraMatrix : " << cameraMatrix << endl;
  cout << "distCoeffs : " << distCoeffs << endl;
  cout << "Rotation vector : " << R << endl;
  cout << "Translation vector : " << T << endl;
 
  return 0;
}