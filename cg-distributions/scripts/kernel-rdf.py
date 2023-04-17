#!/usr/bin/env python
 
import struct
import sys
from numpy import *
from matplotlib.pyplot import*
 
#fid = open('bdf-pair_HH.txt', 'rb')
try:
    fid = open(sys.argv[1],'r')
except:
    print "could not open pairdist file"
    sys.exit(1)

"""
while True:
    start = fid.read(8)
    if start == '':  break
    ts,n = struct.unpack('ii', start)
    print 'reading', n, 'values'
    r = fromfile(fid, dtype=float, count=n)
    steps.append(ts)
    distances.append(r)
"""
steps = []
distances = []
while True:
    line = fid.readline().strip()
    if line == '': break
    if line.startswith('time'):
        ts = int(line.split()[1])
        steps.append(ts)
        n = int(line.split()[-1])
        dist = array([float(fid.readline().strip()) for i in range(n)])
        distances.append(dist)
fid.close()
numstep = len(steps)
rlo = 0.0
rhi = 28.0
dr = 0.1
h = 0.1

r = linspace(rlo+dr/2,rhi-dr/2,rhi/dr)


def hist(x, n):
    bins = [0] * n
    x0 = min(x)
    h = (max(x)-min(x)) / n
    ni = linspace(x0+h/2,max(x)-h/2,n)
    for xi in x:
        b = int((xi - x0) / h)-1
        bins[b] += 1
    return bins,ni

bins,ni = hist(distances[0],500)
bins = array(bins)


def kernel(x, xi, h):
    return sum(exp(-0.5*((x-xi)/h)**2)) / sqrt(2.0*pi)

def kerneldensity(r,dist,h):
    rdf = []
    for ri in r:
        # considering twice the number of distances as we are writing only half
        g  = 2.0*kernel(ri, dist, h) / h
        g /= pi*ri*ri
        rdf.append(g)
    rdf = array(rdf)
    #print max(rdf),min(rdf)
    return rdf

rdf = []
for i in range(numstep):
    rdfi = kerneldensity(r,distances[i],h)
    rdf.append(rdfi)

avgrdf = []
for i in range(len(r)):
    avgi = sum([dg[i] for dg in rdf]) / numstep
    avgrdf.append(avgi)
avgrdf = array(avgrdf)

plot(r,avgrdf)
#show()
savefig(sys.argv[1]+'.png')
clf()
plot(ni,bins)
savefig('hist.png')

    
