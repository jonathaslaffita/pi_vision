
// #include "am7x.h"
// #include "Nonlinear_controller_fcn_control_rf_w_ailerons.h"
// #include "rt_nonfinite.h"
#include <string.h>
// #include "softServo.h"
#include <iostream>
#include <time.h>
// #include "distance_good.hpp"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <chrono>

using namespace std;
using namespace cv;
using namespace chrono;




int main()
// int main()
{

    
      auto start = std::chrono::high_resolution_clock::now();
          
      

        Mat cameraMatrix = (Mat_<double>(3,3) <<  662.81241185, 0.00000, 462.4643793, 0.00000, 664.52073289, 291.17703399, 0.00000, 0.00000, 1.00000     );  //HP

        Mat distCoeffs =  (Mat_<double>(5,1) <<  -0.13983977 , 0.38987987 ,-0.00326027 , 0.00090451, -0.41185575 ); //HP

    // Mat cameraMatrix = (Mat_<double>(3,3) << 1324.41327, 0.00000, 246.59521, 0.00000, 1422.16168, 162.596409, 0.00000, 0.00000, 1.00000); //PI

    // Mat distCoeffs =  (Mat_<double>(5,1) <<  0.9143256 , -1.91459431 ,-0.09495023 ,-0.08486384,  1.62500132);   //PI
       
    vector<vector<Point3f>> objPoints{{Point3f(0, 0, 0),Point3f(6.3, 0, 0), Point3f(6.3, 6.3 ,0),Point3f(0, 6.3, 0)}};//,{Point3f(0.0, 6.0, 0),Point3f(15.0, 6.0, 0), Point3f(15.0, 21.0 ,0),Point3f(0.0, 21.0,0)}};//,{Point3f(0.0, 0.0,0),Point3f(5.0, 0.0,0),Point3f(5.0, 5.0,0),Point3f(0.0, 5.0,0)},{Point3f(10.0, 0.0,0),Point3f(15.0, 0.0,0),Point3f(15.0, 5.0,0),Point3f(10.0, 5.0,0)},{Point3f(0.0, 22.0,0),Point3f(5.0, 22.0,0),Point3f(5.0, 27.0,0),Point3f(0.0, 27.0,0)},{Point3f(10.0, 22.0,0),Point3f(15.0, 22.0,0),Point3f(15.0, 27.0,0),Point3f(10.0, 27.0,0)},{Point3f(6.25, 1.25,0),Point3f(8.75, 1.25,0),Point3f(8.75, 3.625,0),Point3f(6.25, 3.625,0)},{Point3f(6.25, 23.25,0),Point3f(8.75, 23.25,0),Point3f(8.75, 25.75,0),Point3f(6.25, 25.75,0)}}; //{Point3f(-6.3/2, 0, -6.3/2),Point3f(6.3/2, 0, -63./2), Point3f(-6.3/2, 0 ,6.3/2),Point3f(6.3/2, 0,-6.3/2)}, ////{Point3f(0.0, 0.0, 6.0), Point3f(15.0, 0.0, 6.0), Point3f(15.0, 0.0, 21.0), Point3f(0.0, 0.0, 21.0)}, {Point3f(0.0, 0.0, 0.0), Point3f(5.0, 0.0, 0.0), Point3f(5.0, 0.0, 5.0), Point3f(0.0, 0.0, 5.0)}, {Point3f(10.0, 0.0, 0.0), Point3f(15.0, 0.0, 0.0), Point3f(15.0, 0.0, 5.0), Point3f(10.0, 0.0, 5.0)}, {Point3f(0.0, 0.0, 22.0), Point3f(5.0, 0.0, 22.0), Point3f(5.0, 0.0, 27.0), Point3f(0.0, 0.0, 27.0)}, {Point3f(10.0, 0.0, 22.0), Point3f(15.0, 0.0, 22.0), Point3f(15.0, 0.0, 27.0), Point3f(10.0, 0.0, 27.0)}, {Point3f(6.25, 0.0, 1.25), Point3f(8.75, 0.0, 1.25), Point3f(8.75, 0.0, 3.625), Point3f(6.25, 0.0, 3.625)}, {Point3f(6.25, 0.0, 23.25), Point3f(8.75, 0.0, 23.25), Point3f(8.75, 0.0, 25.75), Point3f(6.25, 0.0, 25.75)}}

    //   Ptr<aruco::Dictionary> dictionary =aruco::getPredefinedDictionary(aruco::DICT_4X4_50);  //for pi

      aruco::DetectorParameters detectorParams = aruco::DetectorParameters(); //not for pi
      aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_50); //not for pi
      aruco::ArucoDetector detector(dictionary, detectorParams); //not for pi

      Mat rvecs;
      Mat tvecs;
      Mat prevtvecs = Mat::zeros(3, 1, CV_16S);
      Mat tvecderriv = Mat::zeros(3, 1, CV_16S);
      
      Mat Rmatassume0;
    
          
          Mat image, imageCopy;
        //   inputVideo.retrieve(image);
          image = imread("images/hello3k1.jpeg") ; 
        //   image.copyTo(imageCopy);
          vector<int> ids;
          vector<vector<Point2f>> corners;
          bitwise_not(image,imageCopy);
          detector.detectMarkers(imageCopy, corners, ids); //not for pi
        //   aruco::detectMarkers(imageCopy, dictionary, corners, ids); //for pi
          
          if (ids.size() > 0) {
              cout << "images/hello3k1.jpeg" << endl;
              aruco::drawDetectedMarkers(imageCopy, corners, ids);
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
            cout << corners2 << endl << corners3 << endl      ;
              solvePnPRansac(corners2, corners3, cameraMatrix, distCoeffs, rvecs, tvecs);
                  
              // Draw axis for each marker
              drawFrameAxes(imageCopy, cameraMatrix, distCoeffs, rvecs, tvecs, 10);
              cout << tvecs << endl;             
              tvecs.convertTo(tvecs, CV_16S, 10);

              auto stop = high_resolution_clock::now();
              auto duration = duration_cast<milliseconds>(stop - start);
              cout << "frequency vision " << 1000/duration.count() << "Hz" << endl;
              cout << tvecs << endl;
              // cout << tvecderriv << endl;
              prevtvecs = tvecs;
              start = stop;

              tvecderriv = 1000*(tvecs - prevtvecs)/duration.count();
              tvecderriv.convertTo(tvecderriv, CV_16S, 10);

              //ROTATIONS
              Rodrigues(rvecs, Rmatassume0);
              // cout << "RMAT" << Rmatassume0 << endl;
              
              //////////////////////////////////////
              float sy = sqrt(Rmatassume0.at<double>(0,0) * Rmatassume0.at<double>(0,0) +  Rmatassume0.at<double>(1,0) * Rmatassume0.at<double>(1,0) );
  
              bool singular = sy < 1e-6; // If
          
              float x, y, z;
              if (!singular)
              {
                  x = 57.296 * atan2(Rmatassume0.at<double>(2,1) , Rmatassume0.at<double>(2,2));
                  y = 57.296 * atan2(-Rmatassume0.at<double>(2,0), sy);
                  z = 57.296 * atan2(Rmatassume0.at<double>(1,0), Rmatassume0.at<double>(0,0));
              }
              else
              {
                  x = 57.296 * atan2(-Rmatassume0.at<double>(1,2), Rmatassume0.at<double>(1,1));
                  y = 57.296 * atan2(-Rmatassume0.at<double>(2,0), sy);
                  z = 0;
              }

              cout << "yaw1" <<z<< endl;
          } 
              
            //   undistort(imageCopy, image, cameraMatrix, distCoeffs);
              imwrite("undistorast.jpg", imageCopy); // A JPG FILE IS BEING SAVED
              //////////////////////////////////////////////////////////////
              // int waitTime = 10;
              // // Show resulting image and close window
            //   imshow("out", imageC);
              

          
      }
