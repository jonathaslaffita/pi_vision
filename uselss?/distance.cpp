// Import the aruco module in OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <chrono>

using namespace std;
using namespace cv;
using namespace chrono;


int main()
{
auto start = std::chrono::high_resolution_clock::now();
VideoCapture inputVideo;
inputVideo.open(0);

Mat cameraMatrix = (Mat_<double>(3,3) <<  662.81531922 ,  0.     ,    462.46163531,  0.  ,       664.52337529 ,291.17616957 ,  0.,           0.    ,       1.       );

Mat distCoeffs =  (Mat_<double>(5,1) <<  -0.13983382 , 0.38984575, -0.00326045 , 0.00090355, -0.41180841 );

vector<vector<Point3f>> objPoints{{Point3f(0.0, 6.0, 0),Point3f(15.0, 6.0, 0), Point3f(15.0, 21.0 ,0),Point3f(0.0, 21.0,0)},{Point3f(0.0, 0.0,0),Point3f(5.0, 0.0,0),Point3f(5.0, 5.0,0),Point3f(0.0, 5.0,0)},{Point3f(10.0, 0.0,0),Point3f(15.0, 0.0,0),Point3f(15.0, 5.0,0),Point3f(10.0, 5.0,0)},{Point3f(0.0, 22.0,0),Point3f(5.0, 22.0,0),Point3f(5.0, 27.0,0),Point3f(0.0, 27.0,0)},{Point3f(10.0, 22.0,0),Point3f(15.0, 22.0,0),Point3f(15.0, 27.0,0),Point3f(10.0, 27.0,0)},{Point3f(6.25, 1.25,0),Point3f(8.75, 1.25,0),Point3f(8.75, 3.625,0),Point3f(6.25, 3.625,0)},{Point3f(6.25, 23.25,0),Point3f(8.75, 23.25,0),Point3f(8.75, 25.75,0),Point3f(6.25, 25.75,0)}};

Ptr<aruco::Dictionary> dictionary =aruco::getPredefinedDictionary(aruco::DICT_5X5_50);

Mat rvecs;
Mat tvecs;
Mat prevrvecs;
Mat rvecderriv;
while (inputVideo.grab()) {
    
    Mat image, imageCopy;
    inputVideo.retrieve(image);
    image.copyTo(imageCopy);
    vector<int> ids;
    vector<vector<Point2f>> corners;
    aruco::detectMarkers(image, dictionary, corners, ids);
    
    if (ids.size() > 0) {
        // aruco::drawDetectedMarkers(imageCopy, corners, ids);
        int nMarkers = corners.size();
        vector<Point3f> corners2{};
        vector<Point2f> corners3{};
        for (int i = 0; i < ids.size(); i++) {
        int idx = ids[i];
        corners2.push_back(objPoints[idx][0]);
        corners2.push_back(objPoints[idx][1]);
        corners2.push_back(objPoints[idx][2]);
        corners2.push_back(objPoints[idx][3]);
        corners3.push_back(corners[i][0]);
        corners3.push_back(corners[i][1]);
        corners3.push_back(corners[i][2]);
        corners3.push_back(corners[i][3]);
        }
        
            
        solvePnPRansac(corners2, corners3, cameraMatrix, distCoeffs, rvecs, tvecs);
            
             
        // drawFrameAxes(imageCopy, cameraMatrix, distCoeffs, rvecs, tvecs, 10);
        
        cout << tvecs << endl;
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
        start = stop;
        }

        
    // int waitTime = 10;
    // Show resulting image and close window
    // imshow("out", imageCopy);
    // char key = (char) waitKey(waitTime);
    // if (key == 27)
    //     break;
}
}