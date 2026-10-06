#!/usr/bin/env python3
"""LPI-CGAN Training with RadioML 2018.01A Dataset

The old standalone RadioML trainer was merged into train.py --radioml.
This wrapper exists so existing habits/scripts keep working:

    python train_radioml.py            ==  python train.py --radioml

RadioML mixing: set RADIOML_PATH below (or place RML2018.01A.pkl in this
directory). The mixed AWGN + real-modulated reference signals make the
threat-model detector stronger, which usually improves metric 3.
"""
import sys
import train

if __name__ == '__main__':
    sys.argv = [sys.argv[0], '--radioml'] + sys.argv[1:]
    train.main()
