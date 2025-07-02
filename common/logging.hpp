/********************************************************
 * Author: Namcheol Lee
 * Affiliation: Real-Time Operating System Laboratory, Seoul National University
 * Contact: nclee@redwood.snu.ac.kr
 * Date: 2025-07-02
 * Description: Logging utilities for SoftPreempt
 ********************************************************/

#define LOG_ENABLED 0  // Set to 0 to disable all logging

#if LOG_ENABLED
  #define LOG_INFO(msg)   std::cout << "[INFO] " << msg << std::endl
  #define LOG_ERROR(msg)  std::cerr << "[ERROR] " << msg << std::endl
  #define LOG_RAW(msg)    std::cout << msg << std::endl  // Optional: raw log without prefix
#else
  #define LOG_INFO(msg)
  #define LOG_ERROR(msg)
  #define LOG_RAW(msg)
#endif
