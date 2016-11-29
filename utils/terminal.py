''' author: sam tenka
    credits: I am grateful for the following topical resources: 
        ANSI color codes:
            http://wiki.bash-hackers.org/scripting/terminalcodes
        Sam learned this topic while working on XAI at STR 2016.
    date: 2016-11-20
    descr: Fancy terminal output:via colors,
           progress bars, and standardized
           user-input prompts.
    usage:
        import utils.terminal
'''

from __future__ import print_function
import os
import sys

def flush():
    sys.stdout.flush()

def print_boxed(string, c='#'):
    ''' Print emphasized `string`. '''
    print()
    border_len = 2 + 2*len(c) + len(string)
    border = (c * (border_len//len(c)))[:border_len]
    print(border)
    print('%s %s %s' % (c, string, c))
    print(border)

def colorize(string):
    ''' Replace bracketed color names by ANSI color commands.

        We also support cursor-movement commands.
    '''
    for i, o in {'{RED}':'\033[31m',
                 '{YELLOW}':'\033[33m',
                 '{GREEN}':'\033[32m',
                 '{CYAN}':'\033[36m',
                 '{BLUE}':'\033[34m',
                 '{MAGENTA}':'\033[35m',
                 '{LEFT}':'\033[1D',
                 '{LEFTMOST}':'\033[1024D',
                 '{UP}':'\033[1A'}.items():
        string = string.replace(i, o)
    return string 

def set_color(color):
    ''' Set color of future program output and user input. '''
    print(colorize(color), end='')

def user_input_iterator(prompt='> '):
    ''' Return generator of user input. '''
    colored_prompt = colorize('{BLUE}%s{YELLOW}' % prompt)
    while True:
        ri = raw_input(colored_prompt)
        set_color('{GREEN}')
        if ri=='clear': os.system('clear')
        elif ri=='exit': break
        else: yield ri

def init_progress_bar(size):
    ''' Print initialized progress bar. '''
    print('[%s]' % ('-'*size))

def update_progress_bar(size, progress):
    ''' Print partial progress bar. '''
    set_color('{UP}{LEFTMOST}')
    print('[%s%s%s]' % ('='*(progress), '>' if progress<size else '', '-'*(size-progress-1))) 

def pipe_progress_bar(seq, size=75):
    ''' Pipe a known-length iterator; track using progress bar. '''
    init_progress_bar(size)
    for i, s in enumerate(seq):
        update_progress_bar(size, (size*i)/len(seq))
        yield s
    update_progress_bar(size, size)
