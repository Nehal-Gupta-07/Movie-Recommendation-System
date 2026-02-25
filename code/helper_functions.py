import ast


def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l

def convert5(obj):
    count=0
    l=[]
    for i in ast.literal_eval(obj):
        if count!=5:
            l.append(i['name'])
            count+=1
        else:
            break
    return l

def fetch_director(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l
