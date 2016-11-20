''' author: sam tenka
    date: 2016-11-20
    descr: Load config data
'''

try:
    with open('config.json') as f:
        config = eval(f.read())
except SyntaxError:
    print('Uh oh... I couldn\'t parse the config file. Is it typed correctly? --- utils.config ')
except IOError:
    print('Uh oh... I couldn\'t find the config file. --- utils.config')

def get(attr, root=config):
    ''' Return value of specified configuration attribute. '''
    node = root 
    for part in attr.split('.'):
        node = node[part]
    return node

def test():
    ''' Ensure reading works '''
    assert(get('META.AUTHOR')=='sam tenka')
    assert(get('META.AUTHOR')!='samtenka')
    print('test passed!')
if __name__=='__main__':
    test()
