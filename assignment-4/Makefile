id = chu00071
inputdir = dataset

# Arguments for testing
input = small_gaussian.txt
clusters = 2
blocks = 8
threads = 32

zipped_files = km_cuda.c report.pdf Makefile
cleaned_files = km_cuda km_serial test *.o

# Rules for building programs
all: km_cuda

km_cuda: km_cuda.cu
	nvcc -o km_cuda km_cuda.cu

run: km_cuda
	./km_cuda $(inputdir)/$(input) $(clusters) $(blocks) $(threads)

km_serial: km_serial.c
	gcc -o km_serial km_serial.c

run-serial: km_serial
	./km_serial $(inputdir)/$(input) $(clusters)

test: test.cu
	nvcc -o test test.cu

run-test: test
	./test

# Utility rules
clean:
	rm -f $(cleaned_files)

# Moves outputted data to the plot directory for viewing
move-output:
	mv centroids.txt plot/centroids.txt
	mv clusters.txt plot/clusters.txt

# Zips up files for submission
zip: clean
	mkdir $(id)
	cp $(zipped_files) $(id)
	tar czvf $(id).tar.gz $(id)
	rm -rf $(id)
