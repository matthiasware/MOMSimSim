from tqdm import tqdm
from time import sleep

# for a in tqdm(range(10), desc='foo', leave=None):

#     for b in tqdm(range(10), desc='bar', leave=None):
#         sleep(.1)

#     bar = tqdm(total=10, desc='baz', leave=None)
#     for c in range(10):
#         bar.update(1)
        # sleep(.1)

# for a in tqdm(range(10), desc='0', leave=True, position=0):

#     for b in tqdm(range(10), desc='1', leave=False, position=1):
#         sleep(.1)

#     for b in tqdm(range(10), desc='2', leave=False, position=2):
#         sleep(.1)

for a in range(10):

    for b in tqdm(range(10), desc='1'):
        sleep(.1)

    for b in tqdm(range(10), desc='2'):
        sleep(.1)