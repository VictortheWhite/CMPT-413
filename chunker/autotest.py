#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import answer.perc
import subprocess

for x in os.listdir(os.path.join("model")):
    if x.startswith('model'):
        f = open("output/output" + x[5:], "w")
        subprocess.call(["python3", "answer/perc.py", "-m", "model/" + x], stdout=f)
