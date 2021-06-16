# -*- coding: utf-8 -*-
"""
Created on Sat May  8 12:04:26 2021

@author: mikkel
"""
import math as m
import random as r


def RadialVelocity_DataMaker(a, P, i=90, e=0, sigma=0, n=100):
    list = []
    
    for x in range(n):
        l = 2*m.pi*a*m.sin(m.radians(i))/(P*(1-e**2)**0.5) * (r.uniform(0.95, 1.05)) * m.cos(n)
        
        list.append(l)
    
    return list        

a = 1.5*10**11
P = 365*24*60**2

Data_set = RadialVelocity_DataMaker(a, P, 90, 0, 100)
print(Data_set)
