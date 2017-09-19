#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:54:53 2017

@author: brain_stop
"""

#    def sigma_scaling(self, old_fitness, temp=0.05):
#        mean = np.mean(old_fitness)
#        print("mean", mean)
#        sigma = np.std(old_fitness)
#        print("sigma", sigma)
#        if sigma != 0:
#            new_fitness = (old_fitness - mean) / (2*sigma)
#        else:
#            new_fitness = np.ones(len(old_fitness))
#        return new_fitness

#    def boltzman_scalin(self, old_fitness):
#        mean = np.mean(old_fitness)
#        boltz_min = 1
#        self.boltz_temp -= self.boltz_dt
#        if self.boltz_temp < 1:
#            self.boltz_temp = boltz_min
#        divider = (old_fitness / self.boltz_temp)
#        new_fitnees = divider / (mean / self.boltz_temp)
#        return new_fitnees
