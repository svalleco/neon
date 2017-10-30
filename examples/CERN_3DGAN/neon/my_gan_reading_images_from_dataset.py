import numpy as np
import h5py
import sys
import json

events = []

# if len(sys.argv) != 4:
#   print('Usage: python h5toJSON.py [h5 input file name] [json output file name] [max number of events]')
#   sys.exit()
input_file = "/home/azanetti/CERNDATA/Ele_v1_1_2.h5"
output_file = "/home/azanetti/CERNDATA/Ele_v1_1_2.json"
maxn = 100
ifile = h5py.File(input_file,'r')
print(ifile.keys())
ofile = open(output_file,'w')
# maxn = int(sys.argv[3])


ecal = np.array(ifile['ECAL'])
target = np.array(ifile['target'])

print('TARGET:', target.shape)
print('ECAL:', ecal.shape)

en = 0
ei = ecal.shape[1]
ej = ecal.shape[2]
ek = ecal.shape[3]


ti = target.shape[1]

assert(ecal.shape[0]  == target.shape[0])

nevents = ecal.shape[0]
if ecal.shape[0] > maxn:
  nevents = maxn

for e in range(nevents):

  ec = ecal[e]
  tg = target[e]

  event = {}
  event['primary'] = []
  event['ecal'] = []

  for i in range(ei):
    for j in range(ej):
      for k in range(ek):

        energy = ec[i][j][k]

        if energy > 0:
          event['ecal'].append([i, j, k, energy])


  for i in range(ti):

    energy = tg[i][0]
    px = tg[i][1]
    py = tg[i][2]
    pz = tg[i][3]
    id = tg[i][4]

    event['primary'].append([energy, px, py, pz, id])

  events.append(event)
  en += 1

ofile.write(json.dumps(events))
