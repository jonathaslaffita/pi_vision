#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <math.h>

using namespace std;
using namespace cv;
using namespace aruco;

int main()
{
double time1 = 0.0;
double time2 = 0.0;

// Loading calibration data
FileStorage fs("calibration.xml", FileStorage::READ);
Mat cameraMatrix, distCoeffs;
fs["cameraMatrix"] >> cameraMatrix;
fs["distCoeffs"] >> distCoeffs;
fs.release();

// Defining marker size and object corners
Vec3d MARKER_SIZE(15.0, 5.0, 5.0);
vector<Point3f> Marker_Object_corners;
Marker_Object_corners.push_back(Point3f(0.0, 6.0, 0.0));
Marker_Object_corners.push_back(Point3f(15.0, 6.0, 0.0));
Marker_Object_corners.push_back(Point3f(15.0, 21.0, 0.0));
Marker_Object_corners.push_back(Point3f(0.0, 21.0, 0.0));

// Defining dictionary and detector parameters
Ptr<Dictionary> dictionary = getPredefinedDictionary(DICT_5X5_50);
Ptr<DetectorParameters> detectorParams = DetectorParameters::create();

// Opening video capture object
VideoCapture cap(0);
if (!cap.isOpened())
{
    cout << "Error opening camera." << endl;
    return -1;
}

while (true)
{
    // Capturing frame
    Mat frame;
    cap >> frame;
    if (frame.empty())
        break;

    // Converting frame to grayscale
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    // Detecting markers
    vector<int> markerIds;
    vector<vector<Point2f>> markerCorners, rejectedCandidates;
    detectMarkers(gray, dictionary, markerCorners, markerIds, detectorParams, rejectedCandidates);

    if (markerIds.size() > 0)
    {
        // Initializing object points for one marker
        vector<Point3f> objectPoints;
        for (int i = 0; i < 4; i++)
            objectPoints.push_back(Marker_Object_corners[i]);

        // Initializing image points for one marker
        vector<Point2f> imagePoints = markerCorners[0];

        // Initializing rotation and translation vectors
        Mat rvec, tvec;

        // Solving for pose
        solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

        // Calculating distance
        double distance = sqrt(pow(tvec.at<double>(0, 0), 2) + pow(tvec.at<double>(1, 0), 2) + pow(tvec.at<

jonathaslaffita@gmail.com
a

t<double>(2, 0), 2));
cout << "Distance: " << distance << endl;

        // You may also consider to visualize the detections and the pose of the marker by drawing the axis or the marker's outline on the frame.
        // However, this is out of the scope of this translation and not included in the script.
    }

}

return 0;