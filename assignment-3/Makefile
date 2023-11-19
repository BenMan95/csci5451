id = chu00071
dataset = dataset

# Arguments for testing
input = 1000.txt
output = output/labels.txt
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
	mpirun -np $(threads) ./lpa $(dataset)/$(input) $(output)

# Utility rules
clean:
	rm -f $(cleaned_files)

zip: clean
	mkdir $(id)
	cp $(zipped_files) $(id)
	tar czvf $(id).tar.gz $(id)
	rm -rf $(id)
