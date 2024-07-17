#!/usr/bin/env bash

# python main.py --save-full-performance --aggregator foolsgold --clients 10 --adversary-type onoff_labelflipper --percent-adversaries 0.5 --rounds 5000

for seed in {1..5}; do
  python main.py --save-influence --aggregator foolsgold --clients 10 --adversary-type onoff_freerider --percent-adversaries 0.5 --rounds 5000
  python main.py --save-influence --aggregator foolsgold --clients 10 --adversary-type onoff_freerider --percent-adversaries 0.5 --rounds 5000 --start-on
done
