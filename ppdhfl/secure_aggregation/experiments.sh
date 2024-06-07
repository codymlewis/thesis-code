#!/bin/sh

python main.py --clients 10 --epochs 1 --strategy fedavg
python main.py --clients 10 --epochs 1 --strategy norm
python main.py --clients 10 --epochs 10 --strategy fedavg
python main.py --clients 10 --epochs 10 --strategy norm

python main.py --clients 100 --epochs 1 --strategy fedavg
python main.py --clients 100 --epochs 1 --strategy norm
python main.py --clients 100 --epochs 10 --strategy fedavg
python main.py --clients 100 --epochs 10 --strategy norm
