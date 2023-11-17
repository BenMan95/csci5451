id = chu00071

# Arguments for testing
datadir = dataset
input = 1000.txt
threads = 4

zipped_files = lpa.c utils.c utils.h \
			   report.pdf Makefile

cleaned_files = lpa *.o

# Rules for building programs
all: lpa

lpa: lpa.c utils.o
	mpic++ -o lpa lpa.c utils.o

# Rules for running programs
run: lpa
	mpirun -np $(threads) ./lpa $(input)

# Utility rules
clean:
	rm -f $(cleaned_files)

zip: clean
	mkdir $(id)
	cp $(zipped_files) $(id)
	tar czvf $(id).tar.gz $(id)
	rm -rf $(id)
