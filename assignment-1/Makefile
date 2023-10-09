datadir = dataset
dataset = small
points = $(datadir)/$(dataset)_data.csv
labels = $(datadir)/$(dataset)_label.csv

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

test: test-pthreads test-openmp

test-pthreads: lc_pthreads
	./lc_pthreads $(points) $(labels) $(iters) $(threads)

test-openmp: lc_openmp
	./lc_openmp $(points) $(labels) $(iters) $(threads)

time: time-pthreads time-openmp

time-pthreads:
	make test-pthreads | grep "Execution time"

time-openmp:
	make test-openmp | grep "Execution time"

clean:
	rm -f lc_pthreads lc_openmp lc_serial $(id).tar.gz *.o

zip: clean
	mkdir $(id)
	cp $(files) $(id)
	tar czvf $(id).tar.gz $(id)
	rm -rf $(id)

