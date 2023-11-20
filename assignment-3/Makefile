id = chu00071
inputdir = dataset
outputdir = output

# Arguments for testing
input = 1000.txt
output = labels.txt
procs = 4

zipped_files = lpa.c utils.c utils.h \
			   report.pdf Makefile

cleaned_files = lpa tester *.o

# Rules for building programs
all: lpa tester

lpa: lpa.c utils.o
	mpic++ -o lpa lpa.c utils.o

tester: tester.c
	gcc -o tester tester.c

run: lpa
	mpirun -np $(procs) ./lpa $(inputdir)/$(input) $(outputdir)/$(output)

# Test code only checks to make sure all points which share an edge have the same label
# Will not check to ensure that points which don't share an edge have different labels
test: tester run
	./tester $(inputdir)/$(input) $(outputdir)/$(output)

# Utility rules
clean:
	rm -f $(cleaned_files)

zip: clean
	mkdir $(id)
	cp $(zipped_files) $(id)
	tar czvf $(id).tar.gz $(id)
	rm -rf $(id)
