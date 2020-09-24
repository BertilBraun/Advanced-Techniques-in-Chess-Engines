/****************************************************************************
 * Simple definitions to aid platform portability
 *  Author:  Bill Forster
 *  License: MIT license. Full text of license is in associated file LICENSE
 *  Copyright 2010-2014, Bill Forster <billforsternz at gmail dot com>
 ****************************************************************************/
#ifndef PORTABILITY_H
#define PORTABILITY_H
#include <stdint.h>     // int32_t etc. 
#if defined _WIN32
    #define THC_WINDOWS     // THC = triplehappy chess
#else
    #define THC_UNIX        // pretty good guess hopefully
#endif

#ifdef THC_WINDOWS
   #define WINDOWS_FIX_LATER   // Windows only, fix later on Mac, Linux
#endif

#ifdef THC_WINDOWS
   #include <Windows.h>
   #include <string.h>
   #define strcmpi _strcmpi
#endif

#endif // PORTABILITY_H
