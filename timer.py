# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 07:09:15 2023

@author: JeremyMoss
"""

from contextlib import contextmanager

@contextmanager
my_timer():
    start=time()
    yield
    end=time()
    print("Script completed in", end - start, "seconds")
