#!/bin/bash
gcc helper.c -o libhelper.so -shared -fPIC -O3 -std=c99 -lc -D_POSIX_C_SOURCE=200112 -D_GNU_SOURCE -Wall -pthread
