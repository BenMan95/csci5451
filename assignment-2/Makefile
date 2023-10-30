id = chu00071

workers = 4
size = 1000000

outfile = output.txt
uoutfile = unsorted.txt

zipped_files = qs_mpi.c qs_serial.c report.pdf Makefile
cleaned_files = qs_mpi qs_serial

all: qs_mpi

qs_mpi: qs_mpi.c
	mpic++ -o qs_mpi qs_mpi.c

# Run and save output
run: qs_mpi
	mpirun -np $(workers) ./qs_mpi $(size) $(outfile)

# Time the sorting algorithm without writing output
time: qs_mpi
	mpirun -np $(workers) ./qs_mpi $(size)

time-all: qs_mpi
	for s in 1000000 10000000 100000000; do \
		echo Size: $$s; \
		for w in 1 2 4 8 16; do \
			echo Workers: $$w; \
			make time workers=$$w size=$$s; \
		done; \
	done | grep -E 'Sort Time|Size|Workers'

# Compare unsorted output with that of builtin `sort` command
test: qs_mpi
	mpirun -np $(workers) ./qs_mpi $(size) $(outfile) $(uoutfile)
	diff <(tail -n+2 $(outfile)) <(tail -n+2 $(uoutfile) | sort -n -)

clean:
	rm -f $(cleaned_files)

zip: clean
	mkdir $(id)
	cp $(zipped_files) $(id)
	tar czvf $(id).tar.gz $(id)
	rm -rf $(id)

# Rules to make and run serial implementation
# Not required for assignment, but useful to compare to
qs_serial: qs_serial.c
	gcc -o qs_serial qs_serial.c

run-serial: qs_serial
	./qs_serial $(size) $(outfile)
