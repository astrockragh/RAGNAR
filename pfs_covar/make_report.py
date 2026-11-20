'''
Put pfs_covar/out/report_data.json (analyze.py) and sections_data.json (sections.py) into report_template.html -> pfs_covar/out/pfs_covar_report.html
Run from RAGNAR_new/: python pfs_covar/make_report.py
'''
import os.path as osp
HERE = osp.dirname(osp.abspath(__file__))
data = open(osp.join(HERE, 'out', 'report_data.json')).read()
assert '</script' not in data
sec = open(osp.join(HERE, 'out', 'sections_data.json')).read() # sections.py
assert '</script' not in sec
html = open(osp.join(HERE, 'report_template.html')).read().replace('/*DATA*/', data).replace('/*SECTIONS*/', sec)
open(osp.join(HERE, 'out', 'pfs_covar_report.html'), 'w').write(html)
print(f"pfs_covar/out/pfs_covar_report.html, {len(html)/1e6:.2f} MB")
