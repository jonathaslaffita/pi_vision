#include "am7x.h"
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

mutex mutex_am7;

struct am7_data_out myam7_data_out;
struct am7_data_out myam7_data_out_copy;
struct am7_data_out myam7_data_out_copy_internal;
struct am7_data_in myam7_data_in;
struct am7_data_in myam7_data_in_copy;
float extra_data_in[255], extra_data_in_copy[255];
float extra_data_out[255], extra_data_out_copy[255];
uint16_T buffer_in_counter;
char am7_msg_buf_in[sizeof(struct am7_data_in)*2]  __attribute__((aligned));
char am7_msg_buf_out[sizeof(struct am7_data_out)]  __attribute__((aligned));
uint32_T received_packets = 0, received_packets_tot = 0;
uint32_T sent_packets = 0, sent_packets_tot = 0;
uint32_T missed_packets = 0;
uint16_T sent_msg_id = 0, received_msg_id = 0;
int serial_port;
float ca7_message_frequency_RX, ca7_message_frequency_TX;
struct timeval current_time, last_time, last_sent_msg_time;

int verbose_connection = 1;
int verbose_optimizer = 0;
int verbose_servo = 0; 
int verbose_runtime = 0; 
int verbose_received_data = 0; 

float angle_1_pwm = 1000; 
float angle_2_pwm = 1000; 

// Intitializing variables for vision:
void mainpi(am7_data_out *output);

void am7_init(){

  //Init serial port
  if ((serial_port = serialOpen ("/dev/ttyS0", BAUDRATE_AM7)) < 0){
    fprintf (stderr, "Unable to open serial device: %s\n", strerror (errno)) ;
  }
  if (wiringPiSetup () == -1){
    fprintf (stdout, "Unable to start wiringPi: %s\n", strerror (errno)) ;
  }
  sent_msg_id = 0;

  // //Initialize the servo writing feature
  // softServoSetup (0, 1, 2, 3, 4, 5, 6, 7);

  //Initialize the extra messages value 
  for(int i = 0; i < (sizeof(extra_data_in)/sizeof(float)); i++ ){
    extra_data_in[i] = 0.f;
  }
    for(int i = 0; i < (sizeof(extra_data_out_copy)/sizeof(float)); i++ ){
    extra_data_out_copy[i] = 0.f;
  }
}

void am7_parse_msg_in(){

  mutex_am7.lock();
  memcpy(&myam7_data_in, &am7_msg_buf_in[1], sizeof(struct am7_data_in));
  received_msg_id = myam7_data_in.rolling_msg_in_id;
  extra_data_in[received_msg_id] = myam7_data_in.rolling_msg_in;
  mutex_am7.unlock(); 
}

void writing_routine(){

    mutex_am7.lock();
    memcpy(&myam7_data_out, &myam7_data_out_copy, sizeof(struct am7_data_out));
    memcpy(&extra_data_out, &extra_data_out_copy, sizeof(extra_data_out));
    mutex_am7.unlock();
    myam7_data_out.rolling_msg_out_id = sent_msg_id;
    
    uint8_T *buf_out = (uint8_T *)&myam7_data_out;
    //Calculating the checksum
    uint8_T checksum_out_local = 0;

    for(uint16_T i = 0; i < sizeof(struct am7_data_out) - 1; i++){
      checksum_out_local += buf_out[i];
    }
    myam7_data_out.checksum_out = checksum_out_local;
    //Send bytes
    serialPutchar(serial_port, START_BYTE);
    for(int i = 0; i < sizeof(struct am7_data_out); i++){
      serialPutchar(serial_port, buf_out[i]);
    }
    sent_packets++;
    sent_packets_tot++;
    gettimeofday(&last_sent_msg_time, NULL);

    //Increase the counter to track the sending messages:
    sent_msg_id++;
    if(sent_msg_id == 255){
        sent_msg_id = 0;
    }
}

void reading_routine(){
    uint8_T am7_byte_in;
    while(serialDataAvail(serial_port)){
      am7_byte_in = serialGetchar (serial_port);      
      if( (am7_byte_in == START_BYTE) || (buffer_in_counter > 0)){
        am7_msg_buf_in[buffer_in_counter] = am7_byte_in;
        buffer_in_counter ++;  
      }
      if (buffer_in_counter > sizeof(struct am7_data_in)){
        buffer_in_counter = 0;
        uint8_T checksum_in_local = 0;
        for(uint16_T i = 1; i < sizeof(struct am7_data_in) ; i++){
          checksum_in_local += am7_msg_buf_in[i];
        }
        if(checksum_in_local == am7_msg_buf_in[sizeof(struct am7_data_in)]){
          am7_parse_msg_in();
          received_packets++;
          received_packets_tot++;
        }
        else {
          missed_packets++;
        }
      }
    }
} 

void print_statistics(){
  gettimeofday(&current_time, NULL); 
  if((current_time.tv_sec*1e6 + current_time.tv_usec) - (last_time.tv_sec*1e6 + last_time.tv_usec) > 2*1e6){
    received_packets = 0;
    sent_packets = 0;
    gettimeofday(&last_time, NULL);
    printf("Total received packets = %d \n",received_packets_tot);
    printf("Total sent packets = %d \n",sent_packets_tot);
    printf("Tracking sent message id = %d \n",sent_msg_id);
    printf("Tracking received message id = %d \n",received_msg_id);
    printf("Corrupted packet received = %d \n",missed_packets);
    printf("Average message RX frequency = %f \n",ca7_message_frequency_RX);
    printf("Average message TX frequency = %f \n",ca7_message_frequency_TX);
    fflush(stdout);
  }
  ca7_message_frequency_RX = (received_packets*1e6)/((current_time.tv_sec*1e6 + current_time.tv_usec) - (last_time.tv_sec*1e6 + last_time.tv_usec));
  ca7_message_frequency_TX = (sent_packets*1e6)/((current_time.tv_sec*1e6 + current_time.tv_usec) - (last_time.tv_sec*1e6 + last_time.tv_usec));
}

void send_receive_am7(){

  //Reading routine: 
  reading_routine();
  //Writing routine with protection to not exceed desired packet frequency:
  gettimeofday(&current_time, NULL); 
  if((current_time.tv_sec*1e6 + current_time.tv_usec) - (last_sent_msg_time.tv_sec*1e6 + last_sent_msg_time.tv_usec) > (1e6/MAX_FREQUENCY_MSG_OUT)){
    writing_routine();
  }

  //Print some stats
  if(verbose_connection){
    print_statistics();
  }

}

void* first_thread() //Receive and send messages to pixhawk
{
  while(1){ 
    send_receive_am7();
    // write_to_servos();
  }
  return NULL;
}

// extern "C" void second_thread();
void* second_thread() //Run the optimization code 
{
      
      am7_data_out output1;
      auto start = std::chrono::high_resolution_clock::now();
      VideoCapture inputVideo;
      output1.rolling_msg_out = 0.154687;
      inputVideo.open(0);
      

      Mat cameraMatrix = (Mat_<double>(3,3) <<  662.81531922 ,  0.     ,    462.46163531,  0.  ,       664.52337529 ,291.17616957 ,  0.,           0.    ,       1.       );

      Mat distCoeffs =  (Mat_<double>(5,1) <<  -0.13983382 , 0.38984575, -0.00326045 , 0.00090355, -0.41180841 );
      
      vector<vector<Point3f>> objPoints{{Point3f(0.0, 6.0, 0),Point3f(15.0, 6.0, 0), Point3f(15.0, 21.0 ,0),Point3f(0.0, 21.0,0)},{Point3f(0.0, 0.0,0),Point3f(5.0, 0.0,0),Point3f(5.0, 5.0,0),Point3f(0.0, 5.0,0)},{Point3f(10.0, 0.0,0),Point3f(15.0, 0.0,0),Point3f(15.0, 5.0,0),Point3f(10.0, 5.0,0)},{Point3f(0.0, 22.0,0),Point3f(5.0, 22.0,0),Point3f(5.0, 27.0,0),Point3f(0.0, 27.0,0)},{Point3f(10.0, 22.0,0),Point3f(15.0, 22.0,0),Point3f(15.0, 27.0,0),Point3f(10.0, 27.0,0)},{Point3f(6.25, 1.25,0),Point3f(8.75, 1.25,0),Point3f(8.75, 3.625,0),Point3f(6.25, 3.625,0)},{Point3f(6.25, 23.25,0),Point3f(8.75, 23.25,0),Point3f(8.75, 25.75,0),Point3f(6.25, 25.75,0)}};

      Ptr<aruco::Dictionary> dictionary =aruco::getPredefinedDictionary(aruco::DICT_5X5_50);  //for pi

      // aruco::DetectorParameters detectorParams = aruco::DetectorParameters(); //not for pi
      // aruco::Dictionary dictionary = aruco::getPredefinedDictionary(aruco::DICT_5X5_50); //not for pi
      // aruco::ArucoDetector detector(dictionary, detectorParams); //not for pi

      Mat rvecs;
      Mat tvecs;
      Mat prevrvecs;
      Mat rvecderriv;
      cout << "heloo3" << endl;
      while (inputVideo.grab()) {
          
          Mat image, imageCopy;
          inputVideo.retrieve(image);
          image.copyTo(imageCopy);
          vector<int> ids;
          vector<vector<Point2f>> corners;
          // detector.detectMarkers(image, corners, ids); //not for pi
          aruco::detectMarkers(image, dictionary, corners, ids); //for pi
          
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
                  
              // Draw axis for each marker
                  
              // drawFrameAxes(imageCopy, cameraMatrix, distCoeffs, rvecs, tvecs, 10);
                  
              
              cout << tvecs << endl;
              auto stop = high_resolution_clock::now();
              auto duration = duration_cast<milliseconds>(stop - start);
              cout << "Time taken by function: " << duration.count() << " microseconds" << endl;
              start = stop;
              
          
          // int waitTime = 10;
          // Show resulting image and close window
          // imshow("out", imageCopy);
          // char key = (char) waitKey(waitTime);
          // if (key == 27)
          //     break;

          //Print performances if needed
          // if(verbose_runtime){
          //   printf("\n Elapsed time = %f \n",(float) elapsed_time); 
          //   fflush(stdout);
          // }


          // extra_data_out_copy[0] = 1.453; 
          // extra_data_out_copy[1] = 1.23423; 
        
        output1.pi_translation_x = tvecs.at<int16_t>(0);
        output1.pi_translation_y = tvecs.at<int16_t>(1);
        output1.pi_translation_z = tvecs.at<int16_t>(2);
        output1.pi_rotation_x = rvecs.at<int16_t>(0);
        output1.pi_rotation_y = rvecs.at<int16_t>(1);
        output1.pi_rotation_z = rvecs.at<int16_t>(2);
          }
        else{
        output1.pi_translation_x = 0.0;
        output1.pi_translation_y = 0.0;
        output1.pi_translation_z = 0.0;
        output1.pi_rotation_x = 0.0;
        output1.pi_rotation_y = 0.0;
        output1.pi_rotation_z = 0.0;
        }
          
          mutex_am7.lock();
          memcpy(&myam7_data_out_copy, &output1, sizeof(struct am7_data_out));
          // memcpy(&extra_data_out, &extra_data_out_copy, sizeof(struct am7_data_out));
          mutex_am7.unlock(); 

      }


    //Print received data if needed
    // if(verbose_received_data){

    //   printf("\n ROLLING MESSAGE VARIABLES IN-------------------------------------------------- \n"); 
     
    //   printf(" \n\n\n");

    //   fflush(stdout);
    // }

      
    return NULL;
  }



int main() {

  //Initialize the serial 
  am7_init();

  // make threads
  thread thread1(first_thread);
  thread thread2(second_thread);
  // pthread_create(&thread1, NULL, first_thread, NULL);
  // pthread_create(&thread2, NULL, second_thread, NULL);

  // wait for them to finish
  thread1.join();
  thread2.join(); 

  //Close the serial and clean the variables 
  fflush (stdout);
  close(serial_port);
}
