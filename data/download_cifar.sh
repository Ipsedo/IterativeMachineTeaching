#!/usr/bin/env bash
if ! [ -d "cifar-10-batches-py" ]; then
	wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
	tar xvzf cifar-10-python.tar.gz
	rm -f cifar-10-python.tar.gz
fi