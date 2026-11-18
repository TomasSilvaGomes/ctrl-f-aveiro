#!/usr/bin/env python3
"""Descarrega os 30 ficheiros reais do Wikimedia Commons e cria um ZIP com as fotos.
   Python >=3.9; usa apenas a biblioteca standard. Internet necessária.
   Uso: python descarregar_imagens.py --width 1280
        python descarregar_imagens.py --original
"""
from __future__ import annotations
import argparse, csv, html, json, os, pathlib, shutil, sys, time, urllib.error, urllib.parse, urllib.request, zipfile

BASE = pathlib.Path(__file__).resolve().parent
HEADERS = {'User-Agent':'AveiroUrbanVisualReferences/1.0 (educational municipal dataset audit; https://commons.wikimedia.org/)', 'Accept':'application/json'}

def fetch_bytes(url):
    last=None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=35) as response:
                return response.read()
        except Exception as e:
            last=e
            if attempt < 3: time.sleep(1.5*(attempt+1))
    raise RuntimeError(f'Erro ao descarregar {url}: {last}')

def strip_html(text):
    import re
    return html.unescape(re.sub('<[^>]*>', '', text or '')).strip()

def main():
    ap=argparse.ArgumentParser()
    group=ap.add_mutually_exclusive_group()
    group.add_argument('--width',type=int,default=1280,help='Largura máxima da imagem (1280 por defeito)')
    group.add_argument('--original',action='store_true',help='Descarregar ficheiros originais')
    args=ap.parse_args()
    with (BASE/'fontes.json').open(encoding='utf-8') as f: items=json.load(f)
    manifest=[]; failed=[]
    for item in items:
        title='File:'+item['ficheiro_original']
        params={'action':'query','titles':title,'prop':'imageinfo','iiprop':'url|extmetadata|size','format':'json'}
        if not args.original: params['iiurlwidth']=args.width
        api='https://commons.wikimedia.org/w/api.php?'+urllib.parse.urlencode(params)
        dest=BASE / item['ficheiro_destino']
        try:
            response=json.loads(fetch_bytes(api).decode('utf-8'))
            page=next(iter(response.get('query',{}).get('pages',{}).values()))
            imageinfo=page['imageinfo'][0]
            if page.get('missing') is not None: raise ValueError('Imagem inexistente no Wikimedia Commons')
            imageurl=imageinfo.get('url') if args.original else imageinfo.get('thumburl',imageinfo.get('url'))
            data=fetch_bytes(imageurl)
            if len(data)<2000 or data[:1]==b'<': raise ValueError('O servidor não devolveu uma imagem válida')
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(data)
            meta=imageinfo.get('extmetadata') or {}
            get=lambda key: strip_html(meta.get(key,{}).get('value',''))
            row={**item,'url_imagem':imageurl,'licenca':get('LicenseShortName'),'url_licenca':get('LicenseUrl'),
                 'autor':get('Artist'),'atribuicao':get('Attribution'),'credito':get('Credit'),
                 'restricoes_uso':get('Restrictions'),'estado':'descarregado','bytes':len(data)}
            manifest.append(row)
            print(f"OK {len(manifest)+len(failed):02d}/30  {item['classe']}/{dest.name} ({len(data)//1024} KB)")
        except Exception as e:
            manifest.append({**item,'estado':'erro','erro':str(e)})
            failed.append((title,str(e)))
            print(f'ERRO {title}: {e}',file=sys.stderr)
        time.sleep(0.25)
    fields=list(dict.fromkeys(k for row in manifest for k in row.keys()))
    with (BASE/'LICENCAS_E_FONTES.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(manifest)
    # ZIP pronto a partilhar: apenas imagens realmente obtidas, fontes e instruções.
    output=BASE.parent/'fotografias_urbanas_aveiro_30_imagens.zip'
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in BASE.rglob('*'):
            if path.is_file() and not path.name.endswith('.url'):
                z.write(path,arcname=path.relative_to(BASE))
    print(f'\nFicheiros descarregados: {len(manifest)-len(failed)} de 30')
    print(f'ZIP produzido: {output}')
    if failed:
        print('Alguns ficheiros não foram descarregados; consultar LICENCAS_E_FONTES.csv')
        return 1
    return 0
if __name__=='__main__': sys.exit(main())
