import os
import subprocess as sp
from progress.bar import IncrementalBar as Bar

def countlines(fname):
    p = sp.Popen(['wc', '-l', fname], stdout=sp.PIPE, stderr=sp.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

synsets = open('synsets.txt', 'r')
wnids = []
labels = []
for line in synsets:
    w, l = line.split()
    wnids.append(w)
    labels.append(l)

sp.call(['mkdir','-p','imagenet'])
sp.call(['mkdir','-p','imagenet/images'])
sp.call(['mkdir','-p','imagenet/urls'])
for w in wnids:
    sp.call(['mkdir','-p','imagenet/images/' + w])

print()
bar = Bar('Fetching URL lists...', max=len(wnids))
for i in range(len(wnids)):
    w = wnids[i]
    l = labels[i]
    sp.call(['wget', '-O', 'imagenet/urls/{}.txt'.format(w), 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}'.format(w)], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    bar.next()
print()

timeout = '5'
tries = '3'

for w in wnids:
    print(w)

print()

for w in wnids:
    fname = 'imagenet/urls/{}.txt'.format(w)
    urls = open(fname, 'r')
    bar = Bar('Downloading images for label {}'.format(w), max=countlines(fname))
    for url in urls:
        u = url.strip()
        # print(w, type(w), u, type(u))
        sp.call(['wget', '-T', timeout, '-t', tries, '--directory-prefix', 'imagenet/images/{}'.format(w), u], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        bar.next()
    print()
    urls.close()
    
print('completed')
