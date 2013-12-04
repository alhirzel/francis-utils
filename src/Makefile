# Francis algorithm in C; see README

CC = gcc
CFLAGS = -I/usr/local/include
CFLAGS += -Wall#-pedantic -Wall
LDFLAGS = -L/usr/lib -L/usr/lib64/atlas
LDLIBS = -lgsl -llapack -lblas -lm
COMMON_SOURCES = bulge.c util.c

DEFAULT: test

test: testbench
	./testbench

testbench:        ${COMMON_SOURCES}

clean:
	rm -rf testbench gradient_compute

.PHONY: clean DEFAULT

