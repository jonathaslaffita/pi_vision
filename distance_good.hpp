// //Define the baudrate for the module and the starting byte 
// #define START_BYTE 0x9B  //1st start block identifier byte
// #define BAUDRATE_AM7 460800 //Define the baudrate
// #define MAX_FREQUENCY_MSG_OUT 550 //Define the maximum message output frequency

// //Communication structures
// struct  __attribute__((__packed__)) am7_data_out {
//     //Actuator state
//     int16_t pi_translation_x;
// 	int16_t pi_translation_y;
//     int16_t pi_translation_z;
// 	int16_t pi_rotation_x;
// 	int16_t pi_rotation_y;
// 	int16_t pi_rotation_z;
//     float rolling_msg_in;
//     uint8_t rolling_msg_in_id;
//     uint8_t checksum_in;
// };

// struct  __attribute__((__packed__)) am7_data_in {
    
//     //Motor command
// 	int16_t for_now_nothing;
//     float rolling_msg_out;
//     uint8_t rolling_msg_out_id;
// 	uint8_t checksum_out;
//     };

extern void mainpi(am7_data_out *output);