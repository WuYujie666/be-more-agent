"""Inspect/edit the local memory store while the voice app is stopped."""
import argparse
import json
from memory_backend import MemoryService
from config import CURRENT_CONFIG


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',default=CURRENT_CONFIG.get('memory_dir'))
    sub=p.add_subparsers(dest='operation',required=True)
    sub.add_parser('list')
    sub.add_parser('clear')
    q=sub.add_parser('search');q.add_argument('text')
    for name in ('delete','history','update'):
        q=sub.add_parser(name);q.add_argument('memory_id')
        if name=='update':q.add_argument('text')
    args=vars(p.parse_args());op=args.pop('operation');directory=args.pop('directory')
    m=MemoryService(directory)
    try:
        m.ready.result(40)
        result=m.search(args['text']) if op=='search' else m.submit(op,**args).result(65)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    finally:m.close()

if __name__=='__main__':main()
