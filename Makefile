OUT_DIR = out/

all:

clean:
	find -name *.pyc -delete
	rm -rf $(OUT_DIR)