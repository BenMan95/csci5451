id = chu00071

# Arguments for testing
datadir = dataset
dataset = small
threads = 4

ifeq ($(dataset), small)
	iters = 10
else
	iters = 30
endif

points = $(datadir)/$(dataset)_data.csv
labels = $(datadir)/$(dataset)_label.csv

zipped_files = lc_pthreads.c lc_openmp.c lc_serial.c \
			   utils.c utils.h report.pdf Makefile \
			   small_weights.txt MNIST_weights.txt

cleaned_files = lc_pthreads lc_openmp lc_serial *.o

# Rules for building programs
all: lc_pthreads lc_openmp

lc_pthreads: lc_pthreads.c utils.o
	gcc -o lc_pthreads lc_pthreads.c utils.o

lc_openmp: lc_openmp.c utils.o
	gcc -o lc_openmp lc_openmp.c utils.o -fopenmp

# Rules for running programs
run: run-pthreads run-openmp

run-pthreads: lc_pthreads
	./lc_pthreads $(points) $(labels) $(iters) $(threads)

run-openmp: lc_openmp
	./lc_openmp $(points) $(labels) $(iters) $(threads)

# Rules for timing programs
time: time-pthreads time-openmp

time-pthreads:
	make run-pthreads | grep "Execution time"

time-openmp:
	make run-openmp | grep "Execution time"

# Utility rules
clean:
	rm -f $(cleaned_files)

zip: clean
	mkdir $(id)
	cp $(zipped_files) $(id)
	tar czvf $(id).tar.gz $(id)
	rm -rf $(id)

# Rules for serial implementation
# Not required for assignment, but useful
# to compare with parallel implementation
lc_serial: lc_serial.c utils.o
	gcc -o lc_serial lc_serial.c utils.o

run-serial: lc_serial
	./lc_serial $(points) $(labels) $(iters)

time-serial:
	make run-serial | grep "Execution time"

