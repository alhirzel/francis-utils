# Francis algorithm in C; see README

CC = gcc
CFLAGS = -I/usr/local/include -O3
CFLAGS += -Wall#-pedantic -Wall
LDFLAGS = -L/usr/lib -L/usr/lib64/atlas
LDLIBS = -lgsl -llapack -lblas -lm
COMMON_SOURCES = bulge.c util.c

DEFAULT: grad-test

grad-test: gradient_compute
	./gradient_compute

test: testbench
	./testbench

gradient_compute: ${COMMON_SOURCES}
testbench:        ${COMMON_SOURCES}

clean:
	rm -rf testbench gradient_compute

.PHONY: clean DEFAULT

