# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 14:30:27 2021

@author: El Cocognito
"""
import re
q="can't be a homme-femme?pourquoi? je, ne,sais pas."
q=re.sub(r'[^\w\s\']',' ',q)
print(q)