dataset = small
points = dataset/$(dataset)_data.csv
labels = dataset/$(dataset)_label.csv

ifeq ($(dataset), small)
	iters = 10
else
	iters = 30
endif

threads = 4

id = chu00071
files = lc_pthreads.c lc_openmp.c utils.c utils.h Makefile

all: lc_pthreads lc_openmp

lc_pthreads: lc_pthreads.c utils.o
	gcc -o lc_pthreads lc_pthreads.c utils.o

lc_openmp: lc_openmp.c utils.o
	gcc -o lc_openmp lc_openmp.c utils.o -fopenmp

clean:
	rm -f lc_pthreads lc_openmp lc_serial $(id).tar.gz *.o

zip: clean
	mkdir $(id)
	cp $(files) $(id)
	tar czvf $(id).tar.gz $(id)
	rm -rf $(id)

test: test-pthreads test-openmp

test-pthreads: lc_pthreads
	./lc_pthreads $(points) $(labels) $(iters) $(threads)

test-openmp: lc_openmp
	./lc_openmp $(points) $(labels) $(iters) $(threads)


lc_serial: lc_serial.c utils.o
	gcc -o lc_serial lc_serial.c utils.o

test-serial: lc_serial
	./lc_serial $(points) $(labels) $(iters)

