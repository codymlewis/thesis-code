#!/bin/bash


for allocation in "cyclic" "sim" "full"; do
    echo
    echo "================================================="
    echo "Tabulating for $allocation"

    echo "Majority of data:"
    python tabulate.py -a $allocation

    echo "CIFAR-100 experiments"
    python tabulate.py -a $allocation -r 100 -c 10

    echo "MNIST 100 clients 100% participation"
    python tabulate.py -a $allocation -c 100
    echo "MNIST 100 clients 10% participation"
    python tabulate.py -a $allocation -c 100 -pc 0.1

    echo "N-BAIoT 9 clients"
    python tabulate.py -a $allocation -r 10
    echo "N-BAIoT 90 clients 100% participation"
    python tabulate.py -a $allocation -r 10 -c 90
    echo "N-BAIoT 90 clients 10% participation"
    python tabulate.py -a $allocation -r 10 -c 90 -pc 0.1
    echo "================================================="
    echo
done