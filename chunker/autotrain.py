#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess

for epoch in range(10):
    subprocess.call(["python3", "answer/chunker.py", "-e", "10", "-m", "model/model10_" + str(epoch)])
