id = chu00071
inputdir = dataset

# Arguments for testing
input = small_gaussian.txt
clusters = 3
blocks = 4
threads = 4

# inputfiles = 1000.txt 10000.txt 100000.txt 1000000.txt
# proccounts = 1 2 4 8 16

zipped_files = km_cuda.c report.pdf Makefile
cleaned_files = km_cuda *.o

# Rules for building programs
all: km_cuda

km_cuda: km_cuda.cu
	nvcc -o km_cuda km_cuda.cu

run: km_cuda
	./km_cuda $(inputdir)/$(input) $(clusters) $(blocks) $(threads)

example: example.cu
	nvcc -o example example.cu

run-ex: example
	./example

# time-all: lpa
# 	for input in $(inputfiles); do \
# 		for procs in $(proccounts); do \
# 			echo input=$$input procs=$$procs; \
# 			mpirun -np $$procs ./lpa $(inputdir)/$$input $(outputdir)/$(output); \
# 		done; \
# 	done

# Utility rules
clean:
	rm -f $(cleaned_files)

zip: clean
	mkdir $(id)
	cp $(zipped_files) $(id)
	tar czvf $(id).tar.gz $(id)
	rm -rf $(id)
