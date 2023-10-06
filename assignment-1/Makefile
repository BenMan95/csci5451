dataset = small
iters = 10
threads = 4

points = dataset/$(dataset)_data.csv
labels = dataset/$(dataset)_label.csv

all: lc_pthreads lc_openmp lc_serial

lc_pthreads: lc_pthreads.c utils.o
	gcc -o lc_pthreads lc_pthreads.c utils.o

lc_openmp: lc_openmp.c utils.o
	gcc -o lc_openmp lc_openmp.c utils.o -fopenmp

lc_serial: lc_serial.c utils.o
	gcc -o lc_serial lc_serial.c utils.o

clean:
	rm -f lc_pthreads lc_openmp *.o

test: test-pthreads test-openmp

test-pthreads: lc_pthreads
	./lc_pthreads $(points) $(labels) $(iters) $(threads)

test-openmp: lc_openmp
	./lc_openmp $(points) $(labels) $(iters) $(threads)

test-serial: lc_serial
	./lc_serial $(points) $(labels) $(iters)

test-compare: lc_openmp lc_serial
	./lc_openmp $(points) $(labels) $(iters) $(threads) > output_openmp.txt
	./lc_serial $(points) $(labels) $(iters) > output_serial.txt
