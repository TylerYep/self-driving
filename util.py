import os

def get_run_name():
    dirlist = sorted([f for f in os.listdir('save/') if os.path.isdir(os.path.join('save/', f))])
    dirlist.sort(key=lambda k: (len(k), k)) # Sort alphabetically but by length
    if len(dirlist) == 0:
        result = 'A'
    else:
        last_run_char = dirlist[-1][-1]
        if last_run_char == 'Z':
            result = 'A' * (len(dirlist[-1])+1)
        else:
            result = dirlist[-1][:-1] + chr(ord(last_run_char) + 1)
    os.makedirs(os.path.join('save/', result))
    return result
