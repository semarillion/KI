def TimeToSec(timeString):
    t=[int(a) for a in timeString.split(':')]
    if len(t)==2:
        s =(60,1)
    if len (t)==3:
        s =(3600,60,1)
        
    return sum([a*b for a,b in zip(s,t)])

def Calories(Calo):
    t=Calo.split(',')
    if len(t)==2:
        t = [int(a) for a in Calo.split(',')]
        s = (1000, 1)
        ReturnValue = sum([a * b for a, b in zip(s, t)])
    if len(t)==1:
        ReturnValue = int(Calo)

    return ReturnValue

