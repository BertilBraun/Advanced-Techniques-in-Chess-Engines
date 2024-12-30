# read in in search_tree and search_tree2 and compare line by line, printing the first 100 differing lines

import os

for iter in range(100):
    if not os.path.exists(f'graph_{iter}.dot') or not os.path.exists(f'graph2_{iter}.dot'):
        print(f'File graph_{iter}.dot or graph_{iter}.png does not exist')
        break
    with open(f'graph_{iter}.dot', 'r') as f1, open(f'graph2_{iter}.dot', 'r') as f2:
        print(f'Iteration {iter}:')
        f1_lines = f1.readlines()
        f2_lines = f2.readlines()
        i = 0
        while i < len(f1_lines) and i < len(f2_lines):
            l1 = f1_lines[i]
            l2 = f2_lines[i]
            if l1.strip() != l2.strip():
                print(f'Line {i}: {l1.strip()} != {l2.strip()}')
                # search next [shape=box] and replace it with [shape=box, fillcolor=red, style=filled]
                while i < len(f1_lines):
                    if '[shape=box]' in f1_lines[i]:
                        f1_lines[i] = f1_lines[i].replace('[shape=box]', '[shape=box, fillcolor=red, style=filled]')
                        break
                    i += 1
            i += 1
        with open(f'graph_{iter}_diff.dot', 'w') as f:
            f.writelines(f1_lines)

        os.system(f'dot -Tpng graph_{iter}_diff.dot -o graph_{iter}_diff.png')

        print('Done with iteration', iter)
exit()


diffs = 0

for iter in range(100):
    print(f'Iteration {iter}:')
    if not os.path.exists(f'search_tree_{iter}.txt') or not os.path.exists(f'search_tree2_{iter}.txt'):
        print(f'File search_tree_{iter}.txt or search_tree2_{iter}.txt does not exist')
        continue
    with open(f'search_tree_{iter}.txt', 'r') as f1, open(f'search_tree2_{iter}.txt', 'r') as f2:
        for i, (l1, l2) in enumerate(zip(f1.readlines(), f2.readlines())):
            if l1.strip() != l2.strip():
                print(f'Line {i}: {l1.strip()} != {l2.strip()}')
                diffs += 1
            if diffs == 100:
                exit()  # stop after 100 differences
