#include "am7x.h"
#include <pthread.h>
// #include "Nonlinear_controller_fcn_control_rf_w_ailerons.h"
// #include "rt_nonfinite.h"
#include <string.h>
// #include "softServo.h"
#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>

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

pthread_mutex_t mutex_am7;

int verbose_connection = 1;
int verbose_optimizer = 0;
int verbose_servo = 0; 
int verbose_runtime = 0; 
int verbose_received_data = 0; 

float angle_1_pwm = 1000; 
float angle_2_pwm = 1000; 

// Intitializing variables for vision:


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

  pthread_mutex_lock(&mutex_am7);
  memcpy(&myam7_data_in, &am7_msg_buf_in[1], sizeof(struct am7_data_in));
  received_msg_id = myam7_data_in.rolling_msg_in_id;
  extra_data_in[received_msg_id] = myam7_data_in.rolling_msg_in;
  //Save the pwm for the servos: 
  angle_1_pwm =  (myam7_data_in.pwm_servo_1_int);
  angle_2_pwm =  (myam7_data_in.pwm_servo_2_int);
  pthread_mutex_unlock(&mutex_am7); 
}

void writing_routine(){

    pthread_mutex_lock(&mutex_am7);
    memcpy(&myam7_data_out, &myam7_data_out_copy, sizeof(struct am7_data_out));
    memcpy(&extra_data_out, &extra_data_out_copy, sizeof(extra_data_out));
    pthread_mutex_unlock(&mutex_am7);
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

// void write_to_servos(){

//   softServoWrite (4,  angle_1_pwm - 1000) ;
//   softServoWrite (5,  angle_2_pwm - 1000) ;

//   //Print some stats
//   if(verbose_servo){
//     printf("Servo 1 PWM sent = %f \n",(angle_1_pwm+1000));
//     printf("Servo 2 PWM sent = %f \n",(angle_2_pwm+1000));
//     fflush(stdout);
//   }

// }

void* first_thread() //Receive and send messages to pixhawk
{
  while(1){ 
    send_receive_am7();
    // write_to_servos();
  }
}

void* second_thread() //Run the optimization code 
{

    while(1){ 
    //COPY MEMORY
   
    pthread_mutex_lock(&mutex_am7);   
    memcpy(&myam7_data_in_copy, &myam7_data_in, sizeof(struct am7_data_in));
    memcpy(&extra_data_in_copy, &extra_data_in, sizeof(extra_data_in));
    pthread_mutex_unlock(&mutex_am7);

    //VISION HERE!!!
    


  const float MARKER_SIZE = 15.0; // centimeters
  const Mat MARKER_OBJECT_CORNERS = (Mat_<float>(4, 3) << 0.0, 6.0, 0, 15.0, 6.0, 0, 15.0, 21.0, 0, 0.0, 21.0, 0);

  


    //Do your things with myam7_data_in_copy and extra_data_in_copy as input and myam7_data_out_copy_internal and extra_data_out_copy as outputs

    //Init the variables for the function:
    // double u_out[15];
    // double dv[6];
    // double residuals[6];
    // double elapsed_time; 
    // double N_iterations; 
    // double N_evaluation;
    // double exitflag;

    //Assign variables coming from the communication and physical system properties

	// Float msg variables
  //   float K_p_T = extra_data_in_copy[0], K_p_M = extra_data_in_copy[1], m = extra_data_in_copy[2], I_xx = extra_data_in_copy[3];
  //   float I_yy = extra_data_in_copy[4], I_zz = extra_data_in_copy[5], l_1 = extra_data_in_copy[6], l_2 = extra_data_in_copy[7];
  //   float l_3 = extra_data_in_copy[8], l_4 = extra_data_in_copy[9], l_z = extra_data_in_copy[10];
  //   float max_omega = extra_data_in_copy[11], min_omega = extra_data_in_copy[12], max_b = extra_data_in_copy[13], min_b = extra_data_in_copy[14];
  //   float max_g = extra_data_in_copy[15], min_g = extra_data_in_copy[16], max_theta = extra_data_in_copy[17], min_theta = extra_data_in_copy[18];
  //   float max_alpha = extra_data_in_copy[19]* M_PI/180, min_alpha = extra_data_in_copy[20]* M_PI/180, max_phi = extra_data_in_copy[21];
  //   float Cm_zero = extra_data_in_copy[22], Cm_alpha = extra_data_in_copy[23], Cl_alpha = extra_data_in_copy[24], Cd_zero = extra_data_in_copy[25];
  //   float K_Cd = extra_data_in_copy[26], S = extra_data_in_copy[27], wing_chord = extra_data_in_copy[28], rho = extra_data_in_copy[29];

  //   float W_act_motor_const = extra_data_in_copy[30], W_act_motor_speed = extra_data_in_copy[31]; 
  //   float W_act_tilt_el_const = extra_data_in_copy[32], W_act_tilt_el_speed = extra_data_in_copy[33]; 
  //   float W_act_tilt_az_const = extra_data_in_copy[34], W_act_tilt_az_speed = extra_data_in_copy[35]; 
  //   float W_act_theta_const = extra_data_in_copy[36], W_act_theta_speed = extra_data_in_copy[37]; 
  //   float W_act_phi_const = extra_data_in_copy[38], W_act_phi_speed = extra_data_in_copy[39]; 
  //   float W_dv_1 = extra_data_in_copy[40], W_dv_2 = extra_data_in_copy[41], W_dv_3 = extra_data_in_copy[42];
  //   float W_dv_4 = extra_data_in_copy[43], W_dv_5 = extra_data_in_copy[44], W_dv_6 = extra_data_in_copy[45];
  //   float gamma_quadratic = extra_data_in_copy[46];
	
	//   float Cy_beta = extra_data_in_copy[47], Cl_beta = extra_data_in_copy[48], wing_span = extra_data_in_copy[49];

  //   float speed_aoa_protection = extra_data_in_copy[50];

  //   float W_act_ailerons_const = extra_data_in_copy[51], W_act_ailerons_speed = extra_data_in_copy[52]; 
  //   float min_delta_ailerons = extra_data_in_copy[53], max_delta_ailerons = extra_data_in_copy[54];
  //   float CL_aileron = extra_data_in_copy[55];

   
	// // Real time variables:
  //   double Phi = (myam7_data_in_copy.phi_state_int*1e-2 * M_PI/180);
  //   double Theta = (myam7_data_in_copy.theta_state_int*1e-2 * M_PI/180);
  //   double delta_ailerons = (myam7_data_in_copy.ailerons_state_int*1e-2 * M_PI/180);
  //   double Omega_1 = (myam7_data_in_copy.motor_1_state_int*1e-1), Omega_2 = (myam7_data_in_copy.motor_2_state_int*1e-1);
  //   double Omega_3 = (myam7_data_in_copy.motor_3_state_int*1e-1), Omega_4 = (myam7_data_in_copy.motor_4_state_int*1e-1);
  //   double b_1 = (myam7_data_in_copy.el_1_state_int*1e-2 * M_PI/180), b_2 = (myam7_data_in_copy.el_2_state_int*1e-2 * M_PI/180);
  //   double b_3 = (myam7_data_in_copy.el_3_state_int*1e-2 * M_PI/180), b_4 = (myam7_data_in_copy.el_4_state_int*1e-2 * M_PI/180);
  //   double g_1 = (myam7_data_in_copy.az_1_state_int*1e-2 * M_PI/180), g_2 = (myam7_data_in_copy.az_2_state_int*1e-2 * M_PI/180);
  //   double g_3 = (myam7_data_in_copy.az_3_state_int*1e-2 * M_PI/180), g_4 = (myam7_data_in_copy.az_4_state_int*1e-2 * M_PI/180);
  //   double p = (myam7_data_in_copy.p_state_int*1e-1 * M_PI/180), q = (myam7_data_in_copy.q_state_int*1e-1 * M_PI/180); 
  //   double r = (myam7_data_in_copy.r_state_int*1e-1 * M_PI/180), V = (myam7_data_in_copy.airspeed_state_int*1e-2);
  //   double flight_path_angle = (myam7_data_in_copy.gamma_state_int*1e-2 * M_PI/180);
	//   double Beta = (myam7_data_in_copy.beta_state_int*1e-2 * M_PI/180);
  //   double desired_motor_value = (myam7_data_in_copy.desired_motor_value_int*1e-1), desired_el_value = (myam7_data_in_copy.desired_el_value_int*1e-2 * M_PI/180);
  //   double desired_az_value = (myam7_data_in_copy.desired_az_value_int*1e-2 * M_PI/180), desired_theta_value = (myam7_data_in_copy.desired_theta_value_int*1e-2 * M_PI/180);
  //   double desired_phi_value = (myam7_data_in_copy.desired_phi_value_int*1e-2 * M_PI/180);
  //   double desired_ailerons_value = (myam7_data_in_copy.desired_ailerons_value_int*1e-2 * M_PI/180);
  //   dv[0] = (myam7_data_in_copy.pseudo_control_ax_int*1e-2); dv[1] = (myam7_data_in_copy.pseudo_control_ay_int*1e-2);
  //   dv[2] = (myam7_data_in_copy.pseudo_control_az_int*1e-2); dv[3] = (myam7_data_in_copy.pseudo_control_p_dot_int*1e-1 * M_PI/180);
  //   dv[4] = (myam7_data_in_copy.pseudo_control_q_dot_int*1e-1 * M_PI/180); dv[5] = (myam7_data_in_copy.pseudo_control_r_dot_int*1e-1 * M_PI/180);

    // //// Testing variables
	//
	// // Float msg variables
    // float K_p_T = 9e-06, K_p_M = 1.31e-07, m = 2.35, I_xx = 0.15, I_yy = 0.13, I_zz = 0.2, l_1 = 0.231, l_2 = 0.231, l_3 = 0.39, l_4 = 0.39, l_z = 0;
    // float max_omega = 900, min_omega = 100, max_b = 25, min_b = -130, max_g = 45, min_g = -45, max_theta = 100, min_theta = -30, max_phi = 40;
    // float Cm_zero = 0.01, Cm_alpha = -0.1, Cl_alpha = 4.5, Cd_zero = 0.25, K_Cd = 0.13, S = 0.57, wing_chord = 0.33, rho = 1.225;
	// float W_act_motor_const = 1, W_act_motor_speed = 0, W_act_tilt_el_const = 1, W_act_tilt_el_speed = 0; 
	// float W_act_tilt_az_const = 1, W_act_tilt_az_speed = 1, W_act_theta_const = 400, W_act_theta_speed = -50, W_act_phi_const = 400, W_act_phi_speed = -50;
	// float W_dv_1 = 0.01, W_dv_2 = 0.01, W_dv_3 = 0.01, W_dv_4 = 0.1, W_dv_5 = 0.1, W_dv_6 = 0.01, gamma_quadratic = 1e-5; 
	// float Cy_beta = 0, Cl_beta = 0, wing_span = 1.7;
	//
	// // Real time variables:
	// double Phi = 0, Theta = 0, Psi = 0; 
	// double Omega_1 = 100, Omega_2 = 100, Omega_3 = 100, Omega_4 = 100; 
	// double b_1 = 0 * M_PI/180, b_2 = 0 * M_PI/180, b_3 = 0 * M_PI/180, b_4 = 0 * M_PI/180;
	// double g_1 = 0 * M_PI/180, g_2 = 0 * M_PI/180, g_3 = 0 * M_PI/180, g_4 = 0 * M_PI/180;
	// double p = 0, q = 0, r = 0, V = 0, flight_path_angle = 0 * M_PI/180; 
	// double desired_motor_value = 0, desired_el_value = 0 * M_PI/180, desired_az_value = 0 * M_PI/180, desired_theta_value = 0 * M_PI/180, desired_phi_value = 0 * M_PI/180;
	//
    // dv[0] = 0;
    // dv[1] = 0;
    // dv[2] = -5;
    // dv[3] = 2;
    // dv[4] = 2;
    // dv[5] = 0;
////////////////////////////////// THIS IS NOT USED BY ME
    // Nonlinear_controller_fcn_control_rf_w_ailerons(K_p_T, K_p_M, m, I_xx, I_yy, I_zz, l_1, l_2, l_3, l_4, l_z, 
    //                                          Phi, Theta, Omega_1, Omega_2, Omega_3, Omega_4, b_1, b_2, b_3, b_4, g_1, g_2, g_3, g_4, delta_ailerons, 
    //                                          W_act_motor_const, W_act_motor_speed, W_act_tilt_el_const, W_act_tilt_el_speed, 
    //                                          W_act_tilt_az_const, W_act_tilt_az_speed, W_act_theta_const, W_act_theta_speed, W_act_phi_const, W_act_phi_speed, W_act_ailerons_const, W_act_ailerons_speed,
    //                                          W_dv_1, W_dv_2, W_dv_3, W_dv_4, W_dv_5, W_dv_6,
    //                                          max_omega, min_omega, max_b, min_b, max_g, min_g, max_theta, min_theta, max_phi, max_delta_ailerons, min_delta_ailerons, dv, p, q, r, Cm_zero, Cl_alpha, Cd_zero, K_Cd,
    //                                          Cm_alpha, CL_aileron, rho, V, S, wing_chord, flight_path_angle, max_alpha, min_alpha, Beta, gamma_quadratic,
    //                                          desired_motor_value, desired_el_value, desired_az_value, desired_theta_value, desired_phi_value, desired_ailerons_value, verbose_optimizer, speed_aoa_protection,
    //                                          u_out, residuals, &elapsed_time, &N_iterations, &N_evaluation, &exitflag);
/////////////////////////////////// THIS IS IS NOT USED BY ME

    //Convert the function output into integer to be transmitted to the pixhawk again: 
    // myam7_data_out_copy_internal.motor_1_cmd_int = (int16_T) (u_out[0]*1e1) , myam7_data_out_copy_internal.motor_2_cmd_int = (int16_T) (u_out[1]*1e1);
    // myam7_data_out_copy_internal.motor_3_cmd_int = (int16_T) (u_out[2]*1e1) , myam7_data_out_copy_internal.motor_4_cmd_int = (int16_T) (u_out[3]*1e1);
    // myam7_data_out_copy_internal.el_1_cmd_int = (int16_T) (u_out[4]*1e2*180/M_PI), myam7_data_out_copy_internal.el_2_cmd_int = (int16_T) (u_out[5]*1e2*180/M_PI);
    // myam7_data_out_copy_internal.el_3_cmd_int = (int16_T) (u_out[6]*1e2*180/M_PI), myam7_data_out_copy_internal.el_4_cmd_int = (int16_T) (u_out[7]*1e2*180/M_PI);
    // myam7_data_out_copy_internal.az_1_cmd_int = (int16_T) (u_out[8]*1e2*180/M_PI), myam7_data_out_copy_internal.az_2_cmd_int = (int16_T) (u_out[9]*1e2*180/M_PI);
    // myam7_data_out_copy_internal.az_3_cmd_int = (int16_T) (u_out[10]*1e2*180/M_PI), myam7_data_out_copy_internal.az_4_cmd_int = (int16_T) (u_out[11]*1e2*180/M_PI);
    // myam7_data_out_copy_internal.theta_cmd_int = (int16_T) (u_out[12]*1e2*180/M_PI), myam7_data_out_copy_internal.phi_cmd_int = (int16_T) (u_out[13]*1e2*180/M_PI);
    // myam7_data_out_copy_internal.ailerons_cmd_int = (int16_T) (u_out[14]*1e2*180/M_PI);
    // myam7_data_out_copy_internal.residual_ax_int = (int16_T) (residuals[0]*1e2), myam7_data_out_copy_internal.residual_ay_int = (int16_T) (residuals[1]*1e2);
    // myam7_data_out_copy_internal.residual_az_int = (int16_T) (residuals[2]*1e2), myam7_data_out_copy_internal.residual_p_dot_int = (int16_T) (residuals[3]*1e1*180/M_PI);
    // myam7_data_out_copy_internal.residual_q_dot_int = (int16_T) (residuals[4]*1e1*180/M_PI), myam7_data_out_copy_internal.residual_r_dot_int = (int16_T) (residuals[5]*1e1*180/M_PI);
    // myam7_data_out_copy_internal.n_iteration = (uint16_T) (N_iterations), myam7_data_out_copy_internal.exit_flag_optimizer = (int16_T) (exitflag);
    // myam7_data_out_copy_internal.elapsed_time_us = (uint16_T) (elapsed_time * 1e6), myam7_data_out_copy_internal.n_evaluation = (uint16_T) (N_evaluation);

    //Print received data if needed
    if(verbose_received_data){

      printf("\n ROLLING MESSAGE VARIABLES IN-------------------------------------------------- \n"); 
      // printf("\n K_p_T * 1e-5 = %f \n",(float) K_p_T * 1e5); 
      // printf(" K_p_M * 1e-7 = %f \n",(float) K_p_M * 1e7); 
      // printf(" m = %f \n",(float) m); 
      // printf(" I_xx = %f \n",(float) I_xx); 
      // printf(" I_yy = %f \n",(float) I_yy); 
      // printf(" I_zz = %f \n",(float) I_zz); 
      // printf(" l_1 = %f \n",(float) l_1); 
      // printf(" l_2 = %f \n",(float) l_2); 
      // printf(" l_3 = %f \n",(float) l_3); 
      // printf(" l_4 = %f \n",(float) l_4); 
      // printf(" l_z = %f \n",(float) l_z); 
      // printf(" max_omega = %f \n",(float) max_omega); 
      // printf(" min_omega = %f \n",(float) min_omega); 
      // printf(" max_b = %f \n",(float) max_b); 
      // printf(" min_b = %f \n",(float) min_b); 
      // printf(" max_g = %f \n",(float) max_g); 
      // printf(" min_g = %f \n",(float) min_g); 
      // printf(" max_theta = %f \n",(float) max_theta); 
      // printf(" min_theta = %f \n",(float) min_theta); 
      // printf(" max_alpha = %f \n",(float) max_alpha*180/M_PI); 
      // printf(" min_alpha = %f \n",(float) min_alpha*180/M_PI); 
      // printf(" max_phi = %f \n",(float) max_phi); 
      // printf(" Cm_zero = %f \n",(float) Cm_zero);
      // printf(" Cm_alpha = %f \n",(float) Cm_alpha);
      // printf(" Cl_alpha = %f \n",(float) Cl_alpha);
      // printf(" Cd_zero = %f \n",(float) Cd_zero);
      // printf(" K_Cd = %f \n",(float) K_Cd);
      // printf(" S = %f \n",(float) S);
      // printf(" wing_chord = %f \n",(float) wing_chord);
      // printf(" rho = %f \n",(float) rho);
      // printf(" W_act_motor_const = %f \n",(float) W_act_motor_const);
      // printf(" W_act_motor_speed = %f \n",(float) W_act_motor_speed);
      // printf(" W_act_tilt_el_const = %f \n",(float) W_act_tilt_el_const);
      // printf(" W_act_tilt_el_speed = %f \n",(float) W_act_tilt_el_speed);
      // printf(" W_act_tilt_az_const = %f \n",(float) W_act_tilt_az_const);
      // printf(" W_act_tilt_az_speed = %f \n",(float) W_act_tilt_az_speed);
      // printf(" W_act_theta_const = %f \n",(float) W_act_theta_const);
      // printf(" W_act_theta_speed = %f \n",(float) W_act_theta_speed);
      // printf(" W_act_phi_const = %f \n",(float) W_act_phi_const);
      // printf(" W_act_phi_speed = %f \n",(float) W_act_phi_speed);
      // printf(" W_dv_1 = %f \n",(float) W_dv_1);
      // printf(" W_dv_2 = %f \n",(float) W_dv_2);
      // printf(" W_dv_3 = %f \n",(float) W_dv_3);
      // printf(" W_dv_4 = %f \n",(float) W_dv_4);
      // printf(" W_dv_5 = %f \n",(float) W_dv_5);
      // printf(" W_dv_6 = %f \n",(float) W_dv_6);
      // printf(" gamma_quadratic = %f \n",(float) gamma_quadratic);
      // printf(" Cy_beta = %f \n",(float) Cy_beta);
      // printf(" Cl_beta = %f \n",(float) Cl_beta);
      // printf(" wing_span = %f \n",(float) wing_span);
      // printf(" speed_aoa_protection = %f \n",(float) speed_aoa_protection);
      // printf(" W_act_ailerons_const = %f \n",(float) W_act_ailerons_const);
      // printf(" W_act_ailerons_speed = %f \n",(float) W_act_ailerons_speed);
      // printf(" min_delta_ailerons = %f \n",(float) min_delta_ailerons);
      // printf(" max_delta_ailerons = %f \n",(float) max_delta_ailerons);
      // printf(" CL_aileron = %f \n",(float) CL_aileron);

      // printf("\n REAL TIME VARIABLES IN------------------------------------------------------ \n"); 
      // printf(" Phi_deg = %f \n",(float) Phi*180/M_PI);
      // printf(" Theta_deg = %f \n",(float) Theta*180/M_PI);
      // printf(" delta_ailerons_deg = %f \n",(float) delta_ailerons*180/M_PI);
      // printf(" Omega_1_rad_s = %f \n",(float) Omega_1);
      // printf(" Omega_2_rad_s = %f \n",(float) Omega_2);
      // printf(" Omega_3_rad_s = %f \n",(float) Omega_3);
      // printf(" Omega_4_rad_s = %f \n",(float) Omega_4);
      // printf(" b_1_deg = %f \n",(float) b_1*180/M_PI);
      // printf(" b_2_deg = %f \n",(float) b_2*180/M_PI);
      // printf(" b_3_deg = %f \n",(float) b_3*180/M_PI);
      // printf(" b_4_deg = %f \n",(float) b_4*180/M_PI);
      // printf(" g_1_deg = %f \n",(float) g_1*180/M_PI);
      // printf(" g_2_deg = %f \n",(float) g_2*180/M_PI);
      // printf(" g_3_deg = %f \n",(float) g_3*180/M_PI);
      // printf(" g_4_deg = %f \n",(float) g_4*180/M_PI);
      // printf(" p_deg_s = %f \n",(float) p*180/M_PI);
      // printf(" q_deg_s = %f \n",(float) q*180/M_PI);
      // printf(" r_deg_s = %f \n",(float) r*180/M_PI);
      // printf(" V_m_s = %f \n",(float) V);
      // printf(" flight_path_angle_deg = %f \n",(float) flight_path_angle*180/M_PI);
      // printf(" Beta_deg = %f \n",(float) Beta*180/M_PI);
      // printf(" desired_motor_value_rad_s = %f \n",(float) desired_motor_value);
      // printf(" desired_el_value_deg = %f \n",(float) desired_el_value*180/M_PI);
      // printf(" desired_az_value_deg = %f \n",(float) desired_az_value*180/M_PI);
      // printf(" desired_theta_value_deg = %f \n",(float) desired_theta_value*180/M_PI);
      // printf(" desired_phi_value_deg = %f \n",(float) desired_phi_value*180/M_PI);
      // printf(" desired_ailerons_value_deg = %f \n",(float) desired_ailerons_value*180/M_PI);
      // printf(" dv[0] = %f \n",(float) dv[0]);
      // printf(" dv[1] = %f \n",(float) dv[1]);
      // printf(" dv[2] = %f \n",(float) dv[2]);
      // printf(" dv[3] = %f \n",(float) dv[3]);
      // printf(" dv[4] = %f \n",(float) dv[4]);
      // printf(" dv[5] = %f \n",(float) dv[5]);

      // printf("\n REAL TIME VARIABLES OUT------------------------------------------------------ \n"); 
      // printf(" motor_1_cmd_rad_s = %f \n",(float) u_out[0]);
      // printf(" motor_2_cmd_rad_s = %f \n",(float) u_out[1]);
      // printf(" motor_3_cmd_rad_s = %f \n",(float) u_out[2]);
      // printf(" motor_4_cmd_rad_s = %f \n",(float) u_out[3]);
      // printf(" el_1_cmd_deg = %f \n",(float) u_out[4]*180/M_PI);
      // printf(" el_2_cmd_deg = %f \n",(float) u_out[5]*180/M_PI);
      // printf(" el_3_cmd_deg = %f \n",(float) u_out[6]*180/M_PI);
      // printf(" el_4_cmd_deg = %f \n",(float) u_out[7]*180/M_PI);
      // printf(" az_1_cmd_deg = %f \n",(float) u_out[8]*180/M_PI);
      // printf(" az_2_cmd_deg = %f \n",(float) u_out[9]*180/M_PI);
      // printf(" az_3_cmd_deg = %f \n",(float) u_out[10]*180/M_PI);
      // printf(" az_4_cmd_deg = %f \n",(float) u_out[11]*180/M_PI);
      // printf(" theta_cmd_deg = %f \n",(float) u_out[12]*180/M_PI);
      // printf(" phi_cmd_deg = %f \n",(float) u_out[13]*180/M_PI);
      // printf(" ailerons_cmd_deg = %f \n",(float) u_out[14]*180/M_PI);
      // printf(" residuals_ax = %f \n",(float) residuals[0]);
      // printf(" residuals_ay = %f \n",(float) residuals[1]);
      // printf(" residuals_az = %f \n",(float) residuals[2]);
      // printf(" residuals_p_dot_deg_s = %f \n",(float) residuals[3]*180/M_PI);
      // printf(" residuals_q_dot_deg_s = %f \n",(float) residuals[4]*180/M_PI);
      // printf(" residuals_r_dot_deg_s = %f \n",(float) residuals[5]*180/M_PI);
      // printf(" N_iterations = %d \n",(int) N_iterations);
      // printf(" exit_flag_optimizer = %d \n",(int) exitflag);
      // printf(" elapsed_time_uS = %d \n",(int) (elapsed_time * 1e6));
      // printf(" N_evaluations = %d \n",(int) N_evaluation);
      printf(" \n\n\n");

      fflush(stdout);
    }

    //Print performances if needed
    if(verbose_runtime){
      printf("\n Elapsed time = %f \n",(float) elapsed_time); 
      fflush(stdout);
    }

    //Assign the rolling messages to the appropriate value [dummy]
    extra_data_out_copy[0] = 1.453; 
    extra_data_out_copy[1] = 1.23423; 
    
    pthread_mutex_lock(&mutex_am7);
    memcpy(&myam7_data_out_copy, &myam7_data_out_copy_internal, sizeof(struct am7_data_out));
    memcpy(&extra_data_out, &extra_data_out_copy, sizeof(struct am7_data_out));
    pthread_mutex_unlock(&mutex_am7);    
  }
}

void main() {

  //Initialize the serial 
  am7_init();

  pthread_t thread1, thread2;

  // make threads
  pthread_create(&thread1, NULL, first_thread, NULL);
  pthread_create(&thread2, NULL, second_thread, NULL);

  // wait for them to finish
  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL); 

  //Close the serial and clean the variables 
  fflush (stdout);
  close(serial_port);
}
