all:
	pushd libs/nms/; /usr/local/python-3.6.5/bin/python3 setup_linux.py build_ext --inplace; pushd
clean:
	pushd libs/nms/; rm *.so *.c *.cpp; rm -r build; pushd
