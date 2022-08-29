from tqdm import tqdm
import time, random


    
try:
    with tqdm(range(10)) as t:
        for i in t:
            time.sleep(random.random())
except KeyboardInterrupt:
    t.close()
    raise
t.close()