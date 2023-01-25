#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>
#include <termios.h>
#include <wiringPi.h>
#include <wiringSerial.h>
#include <sys/time.h>
#include <math.h>
#include "rtwtypes.h"
#include <stddef.h>
#include <mutex>

#ifndef AM7_H
#define AM7_H

extern std::mutex mutex_am7;

//Define the baudrate for the module and the starting byte 
#define START_BYTE 0x9B  //1st start block identifier byte
#define BAUDRATE_AM7 460800 //Define the baudrate
#define MAX_FREQUENCY_MSG_OUT 550 //Define the maximum message output frequency

//Communication structures
struct  __attribute__((__packed__)) am7_data_out {
    //Actuator state
    int16_t pi_translation_x;
	int16_t pi_translation_y;
    int16_t pi_translation_z;
	int16_t pi_rotation_x;
	int16_t pi_rotation_y;
	int16_t pi_rotation_z;
    float rolling_msg_out;
    uint8_T rolling_msg_out_id;
	uint8_T checksum_out;
};

struct  __attribute__((__packed__)) am7_data_in {
    
    //Motor command
	int16_t for_now_nothing;
    float rolling_msg_in;
    uint8_t rolling_msg_in_id;
    uint8_t checksum_in;
    
    };

#endif

