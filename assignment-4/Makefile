id = chu00071
inputdir = dataset

# Arguments for testing
input = small_gaussian.txt
clusters = 2
blocks = 2
threads = 32

zipped_files = km_cuda.cu km_serial.c report.pdf Makefile
cleaned_files = km_cuda km_serial *.o

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

time-all:
	./km_cuda $(inputdir)/large_cpd.txt 256 2 32
	./km_cuda $(inputdir)/large_cpd.txt 256 4 32
	./km_cuda $(inputdir)/large_cpd.txt 256 2 64
	./km_cuda $(inputdir)/large_cpd.txt 256 4 64
	./km_cuda $(inputdir)/large_cpd.txt 512 4 32
	./km_cuda $(inputdir)/large_cpd.txt 512 8 32
	./km_cuda $(inputdir)/large_cpd.txt 512 4 64
	./km_cuda $(inputdir)/large_cpd.txt 512 8 64
	./km_cuda $(inputdir)/large_cpd.txt 1024 4 32
	./km_cuda $(inputdir)/large_cpd.txt 1024 8 32
	./km_cuda $(inputdir)/large_cpd.txt 1024 16 32
	./km_cuda $(inputdir)/large_cpd.txt 1024 4 64
	./km_cuda $(inputdir)/large_cpd.txt 1024 8 64
	./km_cuda $(inputdir)/large_cpd.txt 1024 16 64
	./km_cuda $(inputdir)/large_cpd.txt 1024 4 128
	./km_cuda $(inputdir)/large_cpd.txt 1024 8 128
	./km_cuda $(inputdir)/large_cpd.txt 1024 16 128

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
