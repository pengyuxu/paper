''' 
/* SHREC 2025
Marco Guerra

*/
'''

# utilities

import json
from pathlib import Path

def write_jsonl( file : str | Path , params : dict ):
    ''' Append to json file describing a run, either of descriptor
    computation or learning training

    PARAMS:
    file : str | Path - The path of the file to append to
    params : dict - A dictionary of parameters of the run
    OUTPUT:
    
    '''

    with open(file, "a+") as f:
        json.dump(params, f)
        f.write("\n")


def get_free_id( file : str | Path ):
    ''' Read jsonl file to check what is the next available id

    PARAMS
    file : str | Path - json file to read

    OUTPUT:
    free_id : int - The first available id
    '''

    lines = []
    try:
        f = open(file, 'r')

    except OSError: # No file, so all free
        return 0

    else: # file found
        with f:
            for line in f:
                lines.append( json.loads(line) ) # load string

    ids = [ x['id'] for x in lines ]

    try:
        last_used_id = max(ids)

    except ValueError: # empty file
        return 0

    free_id = last_used_id + 1
    return free_id

def clear_jsonl(file : str | Path):
    ''' Empty jsonl file

    '''
    try:
        f = open(file, 'r+')

    except OSError: # No file, raise error
        raise ValueError

    else:
        f.truncate(0) # need '0' when using r+

    
