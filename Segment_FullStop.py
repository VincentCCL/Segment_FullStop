#Segment_FullStop.py
#--------------------

# pip install deepmultilingualpunctuation

import click
import sys
import re
import pickle
from collections import deque
from functools import partial
from itertools import product
from deepmultilingualpunctuation import PunctuationModel

@click.command()
@click.option('--w',default=200,help='window size for punctuation prediction')
@click.option('--m', default='sonar', help="dataset on which model is trained: ep|sonar|multi|multisonar")
@click.option('--t', default=0.1, help="threshold for accepting punctuation")
@click.argument('name')

def options(w,m,t,name):
    print('Window size',w,file=sys.stderr)
    if m=='sonar':
        model="oliverguhr/fullstop-dutch-sonar-punctuation-prediction"
    elif m=='ep':
        model="oliverguhr/fullstop-dutch-punctuation-prediction"
    elif m=='multi':
        model="oliverguhr/fullstop-punctuation-multilingual-base"
    elif m=='multisonar':
        model="oliverguhr/fullstop-punctuation-multilingual-sonar-base"
    print('Model',model,file=sys.stderr)
    print('name',name,file=sys.stderr)
    print('threshold',t,file=sys.stderr)
    main(w,t,model,name)




def Segment(tokens,name,max_length,threshold,model): 
    to_translate=[]
    to_translate_indices=[]
    i=0
    stepsize=1
    max_counter={}
    punctuation_counter={}
    lengte=len(tokens)
    while i < len(tokens):
        if i+max_length > len(tokens):
            max=len(tokens)
        else: max=i+max_length
        sourcelist=tokens[i:max]
        to_translate_indices.append([i,max-1])
        to_translate.append(sourcelist)
        i+=stepsize
    #pickle_filename=name+'.translated.pcl'
    print("Segmenting",lengte,"segments",file=sys.stderr)
    results=[]
    nr=0
    nrtotranslate=len(to_translate)
    for text in to_translate:
        nr+=1
        print("Inserting punctuation",nr,'/',nrtotranslate,file=sys.stderr,end='\r')
        try: results.append(model.restore_punctuation(' '.join(text)))  # if punctuator fails, add text without punctuation
        except: results.append(' '.join(text))
    segment_endings=[]
    for t in range(len(to_translate)):
        print("Investigating token",t,'/',lengte,file=sys.stderr, end='\r')
        sourcetext=to_translate[t]
        targettext=results[t].split()
        for i in range(len(sourcetext)):
            if  i < len(targettext) and (sourcetext[i]+'.' == targettext[i] or sourcetext[i]+'?' == targettext[i]):
                segmentposition=t+i
                if not segmentposition in punctuation_counter:
                    punctuation_counter[segmentposition]=1
                else:
                    punctuation_counter[segmentposition]+=1
                sourcetext[i]=targettext[i]
    # Add punctuation to previous token
    for (key,value) in sorted(punctuation_counter.items()):
        if value/max_length > threshold:
            token=tokens[key]
            tokens[key] = token+' .'
            segment_endings.append(key)
    segment_endings.append(len(tokens)-1) # for last segment
    # Create Segments
    prev_ending=-1
    segments=[]
    for segment_ending in segment_endings:
        segments.append([prev_ending+1,segment_ending])
        prev_ending=segment_ending
    return segments#,segnr

def main(max_length,threshold,modelname,inputfile):
    #max_length=200   
    #threshold=0.1 # threshold of ratio of punctuation prediction 

    model = PunctuationModel(model=modelname)

    #inputfile=sys.argv[1]        
    tokens=[]
    infile=open(inputfile,'r')
    lines=infile.readlines()
    for line in lines:
        tokens.extend(line.split())
    
    segments=Segment(tokens,inputfile,max_length,threshold,model)  
    for segment in segments:
        sentence=tokens[segment[0]:segment[1]+1]
        print(' '.join(sentence))
    pass


if __name__=='__main__':
    options()

    
