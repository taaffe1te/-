import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy
import math
import os

TASK1_1 = True
TASK1_2 = False
TASK1_3 = False
TASK1_4 = False
TASK2 = False

R1 = 11
G1 = 10
B1 = 11
R2 = 10
G2 = 9
B2 = 10
R3 = 9
G3 = 11
B3 = 5

def save(name='', fmt='png'):
    #return
    pwd = os.getcwd()
    iPath = './{}'.format(fmt)
    if not os.path.exists(iPath):
        os.mkdir(iPath)
    os.chdir(iPath)
    plt.savefig('{}.{}'.format(name, fmt), dpi=150, bbox_inches='tight')
    os.chdir(pwd)

def fac(x) :
    q = 1
    while x > 1 :
        q *= x
        x -= 1

    return q

def C(k, n) :
    return fac(n)/(fac(k)*fac(n-k))

def diaposon(x, y, c) :
    q = [x]
    while (x < y) :
        x += c
        q.append(x)

    return q

def f(P, i) :
    return P**i/fac(i)

COUNT = 1

def plot(y, x, l, lh, name, lf='', show=True, lw=1.5) :
    global COUNT
    plt.figure(figsize=(16, 8))
    #print(x, len(y[0]))
    for i in range(len(x)) :
        if i == 10 : lf = lf.replace('o', '^')
        plt.plot(x[i], y[i], lf, alpha=0.7, label=l[i], lw=lw, mec='b', mew=2, ms=10)
    plt.ylabel(lh[0])
    plt.xlabel(lh[1])
    plt.title(name)
    plt.grid()
    #plt.margins(0)
    
    if show :
        plt.legend()
        #save(name=name, fmt='pdf')
        save(name=name + ' (' + str(COUNT) + ')', fmt='png')
        COUNT += 1
        plt.show()

def plotSaveShow(name) :
    save(name=name, fmt='png')
    plt.show()

def t1_1(name) :
    Tc = R1
    Ts = R1+G1+B1+R2+G2+B2

    lam = 1/Tc
    u = 1/Ts
    p = lam/u
    print('p =', p)

    n = 1
    Prej = 1

    Prej_ = []
    M_ = []
    K_ = []
    n_ = []
    
    while (Prej > 0.01) :
        q = 0
        for i in range(0, n+1) :
            q += f(p, i)
        P0 = 1/q

        Prej = f(p, n)*P0

        M = 0
        for i in range(0, n+1) :
            M += f(p, i)*P0*i

        K = M/n

        Prej_.append(Prej)
        M_.append(M)
        K_.append(K)
        n_.append(n)

        n += 1

        print([Prej], n)
    
    plot([Prej_], [n_], ['P (отказа)'], ['P (отказа)', 'n'], name+' P (отказа)', '--o')
    plot([M_], [n_], ['M'], ['M', 'n'], name+' M', '--o')
    plot([K_], [n_], ['k'], ['k', 'n'], name+' k', '--o')

def Pi1_2(p, i, P0) :
    return f(p, i) * P0

def Pni1_2(p, i, P0, n) :
    #return p**i/fac(n) * p**i/n**i * P0
    return p**(n+i)/fac(n) / n**i * P0

def kilme(p, i, k) :
    return p**(i+k)/fac(i)/i**k

def miku(n, m, l, u) :
    p = l/u
    
    qn = 0
    for i in range(0, n+1) :
        qn += l**i / fac(i) / u**i
    qm = 0
    for i in range(1, m+1) :
        qm += (l/n/u)**i
    qm *= l**n / fac(n) / u**n
    #print(qm, qn)
    P0 = 1 / (qn + qm)

    Prej = l**n / fac(n) / u**n * (l/n/u)**m * P0
    #print(n, m, Prej)

    Mn = 0
    for i in range(0, n+1) :
        Mn += p**i / fac(i) * P0 * i
    Mm = 0
    for j in range(1, m+1) :
        Mm += Pni1_2(p, j, P0, n) * n
    M = Mn + Mm

    K = M/n

    #if n == 1 or n == 2 : print(n, m, M)

    Pseq = 0
    for i in range(1, m+1) :
        Pseq += Pni1_2(p, i, P0, n)

    Mml = 0
    for i in range(1, m+1) :
        Mml += Pni1_2(p, j, P0, n) * i
    Mml = p**(n+1)/n/fac(n) * (1-(p/n)**m*(m+1-m/n*p))/(1-p/n)**2*P0

    Km = Mml/m

    return Prej, M, K, Pseq, Mml, Km

def t1_2(name) :    
    Tc = R1
    Ts = R1+G1+B1+R2+G2+B2

    Tc = 5
    Ts = 43

    lam = 1/Tc
    u = 1/Ts
    p = lam/u

    Prej__ = []
    M__ = []
    K__ = []
    Pseq__ = []
    Mml__ = []
    Km__ = []

    #n_ = [i for i in range(1, 11)]
    n_ = diaposon(1, 20, 1)
    m_ = diaposon(1, 20, 1)

    m = diaposon(1, 20, 1)
    for n in n_ :
        Prej_ = []
        M_ = []
        K_ = []
        Pseq_ = []
        Mml_ = []
        Km_ = []
    
        for i in m :
            #Prej, M, K, Pseq, Mml, Km = aaaaaaaaaaaa(n, i, p)
            Prej, M, K, Pseq, Mml, Km = miku(n, i, lam, u)
            Prej_.append(Prej)
            M_.append(M)
            K_.append(K)
            #M_.append(-Prej/3.8+0.26923)
            #K_.append((-Prej/3.8+0.26923)/n)
            Pseq_.append(Pseq)
            Mml_.append(Mml)
            Km_.append(Km)
        
        Prej__.append(Prej_)
        M__.append(M_)
        K__.append(K_)
        Pseq__.append(Pseq_)
        Mml__.append(Mml_)
        Km__.append(Km_)

    names = createNames('P (отказа) n = ', n_)
    
    plot(Prej__, [m]*len(n_), names, ['P (отказа)', 'm']*len(n_), name+' P (отказа)', '--o', False)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=4, fancybox=True, shadow=True)
    plotSaveShow(name+' P (отказа)')

    names = createNames('M (оп) n = ', n_)
    plot(M__, [m]*len(n_), names, ['M (операторов)', 'm']*len(n_), name+' M (операторов)', '--o')

    names = createNames('k (оп), n = ', n_)
    plot(K__, [m]*len(n_), names, ['k (операторов)', 'm']*len(n_), name+' k (операторов)', '--o')

    names = createNames('P (оч) n = ', n_)
    plot(Pseq__, [m]*len(n_), names, ['P (очереди)', 'm']*len(n_), name+' P (очереди)', '--o', False)
    plt.legend(loc='upper center', bbox_to_anchor=(0.90, 0.70),
          ncol=2, fancybox=True, framealpha=0.8)
    plotSaveShow(name+' P (очереди)')

    names = createNames('M (очереди) n = ', n_)
    plot(Mml__, [m]*len(n_), names, ['M (очереди)', 'm']*len(n_), name+' M (очереди)', '--o')

    names = createNames('k (оч) n = ', n_)
    plot(Km__, [m]*len(n_), names, ['k (очереди)', 'm']*len(n_), name+' k (очереди)', '--o', False)
    plt.legend(loc='upper center', bbox_to_anchor=(0.90, 0.70),
          ncol=2, fancybox=True, framealpha=0.8)
    plotSaveShow(name+' k (очереди)')

def t1_22(name) :
    Tc = R1
    Ts = R1+G1+B1+R2+G2+B2

    lam = 1/Tc
    u = 1/Ts
    p = lam/u

    Prej__ = []
    M__ = []
    K__ = []
    Pseq__ = []
    Mml__ = []
    Km__ = []

    
    n_ = diaposon(1, 20, 1)
    m_ = diaposon(1, 20, 1)

    n = diaposon(1, 20, 1)
    for m in m_ :
        Prej_ = []
        M_ = []
        K_ = []
        Pseq_ = []
        Mml_ = []
        Km_ = []
    
        for i in n :
            Prej, M, K, Pseq, Mml, Km = miku(i, m, lam, u)
            Prej_.append(Prej)
            M_.append(M)
            K_.append(K)
            #M_.append(-Prej/3.8+0.26923)
            #K_.append((-Prej/3.8+0.26923)/i)
            Pseq_.append(Pseq)
            Mml_.append(Mml)
            Km_.append(Km)
        
        Prej__.append(Prej_)
        M__.append(M_)
        K__.append(K_)
        Pseq__.append(Pseq_)
        Mml__.append(Mml_)
        Km__.append(Km_)

    names = createNames('P (отказа) m = ', m_)
    plot(Prej__, [n]*len(m_), names, ['P (отказа)', 'm']*len(n_), name+' P (отказа)', '--o')

    names = createNames('M (операторов) m = ', m_)
    plot(M__, [n]*len(m_), names, ['M (операторов)', 'm']*len(n_), name+'M (операторов)', '--o')

    names = createNames('k (операторов), m = ', m_)
    plot(K__, [n]*len(m_), names, ['k (операторов)', 'm']*len(n_), name+' k (операторов)', '--o')

    names = createNames('P (очереди) m = ', m_)
    plot(Pseq__, [n]*len(m_), names, ['P (очереди)', 'm']*len(n_), name+' P (очереди)', '--o')

    names = createNames('M (очереди) m = ', m_)
    plot(Mml__, [n]*len(m_), names, ['M (очереди)', 'm']*len(n_), name+' M (очереди)', '--o')

    names = createNames('k (очереди) m = ', m_)
    plot(Km__, [n]*len(m_), names, ['k (очереди)', 'm']*len(n_), name+' k (очереди)', '--o')

def t1_4(name) :
    qn = 0
    for i in range(0, n+1) :
        qn += f(p, i)
    qm = 0
    n = qm+2
    for i in range(1, m+1) :
        qm += p**i/n**i
    qm *= f(p, n)
    P0 = 1/(qn + qm)
    m = P0*2

    Prej = f(p, n) * p**m/n**m *P0


    Mn = 0
    for i in range(0, qn+1) :
        Mn += Pi1_2(p, i, P0) * i
    Mm = 0
    for j in range(1, m+1) :
        Mm += Pni1_2(p, j, P0, n) * n
    M = Mn + Mm

    K = M/n

    Pseq = 0
    Pbe = Mn
    for i in range(0, m+1) :
        Pseq = f(p, n)*((1-f(p, n)**i)/(1-p/n))*P0
    #Pseq = f(p, n)*((1-f(p, n)**m)/(1-p/n))*P0

    Mml = 0
    for i in range(0, m+1) :
        Mml += p**i/fac(n)*p**i/n**i*P0 * i

    Km = Mml/m
    
    plot([Mn], [[1, 2, 3]], ['Mn'], ['Mn', 'm'], name+'_20_Mn', '--o')
    plot([Kn], [[1, 2, 3]], ['Kn'], ['Kn', 'm'], name+'_21_Kn', '--o')
    plot([Pbe], [[1, 2, 3]], ['Pbeing'], ['Pbeing', 'm'], name+'_22_Pbeing', '--o')
    plot([Mm], [[1, 2, 3]], ['Mm'], ['Mm', 'm'], name+'_23_Mm', '--o')

def yamero(n, lam, u) :
    p = lam/u
    
    M = min(lam/u, n)

    K = min(M / n, 1)

    P0 = 1 + p**(n+1)/fac(n)/(n-p)
    for i in range(1, n+1) : P0 += p**i / fac(i)
    P0 = 1/P0

    Pq = min(p**(n+1)/fac(n)/(n-p) * P0, 1)

    Mq = max(p**(n+1)/n/fac(n)/(1-p/n)**2 * P0, 0)

    print(Mq)
    
    return M, K, Pq, Mq

def Miku(name) :
    Tc = R1
    Ts = R1+G1+B1+R2+G2+B2

    Tc = 5
    Ts = 43

    lam = 1/Tc
    u = 1/Ts
    p = lam/u

    M = []
    K = []
    Pq = []
    Mq = []
    
    for n in range(1, 21) :
        M_, K_, Pq_, Mq_ = yamero(n, lam, u)

        M.append(M_)
        K.append(K_)
        Pq.append(Pq_)
        Mq.append(Mq_)

    n = diaposon(1, 20, 1)
    plot([M], [n], ['M (операторов)'], ['M (операторов)', 'n'], name+' M (операторов)', '--o')
    plot([K], [n], ['k (операторов)'], ['k (операторов)', 'n'], name+' k (операторов)', '--o')
    plot([Pq], [n], ['P (очереди)'], ['P (очереди)', 'n'], name+' P (очереди)', '--o')
    plot([Mq[8:]], [n[8:]], ['M (очереди)'], ['M (очереди)', 'n'], name+' M (очереди)', '--o')

def forMoon(n, lam, u, j, v) :
    p = 1
    for i in range(1, j+1) :
        p *= lam/(n*u + i*v)

    return p

def moon(n, lam, u, Tw, inf) :
    p = lam/u

    v = 1/Tw
    
    s1 = 0
    for i in range(n+1) :
        s1 += lam**i / (fac(i)*u**i)
    s2 = 0
    for i in range(1, inf) :
        s2 += forMoon(n, lam, u, i, v)
    
    P0 = 1 / (s1 + lam**n/fac(n)/u**n*s2)
    

    Pn = P0
    for i in range(1, n+1) :
        Pn *= p/i


    s1 = 0
    for i in range(n+1) :
        s1 += i*lam**i / (fac(i)*u**i)

    s2 = 0
    for i in range(1, inf) :
        s2 += forMoon(n, lam, u, i, v)

    M = P0*s1 + n*Pn*s2
    
    K = M/n

    Pq = Pn*s2

    s2 = 0
    for i in range(1, inf) :
        s2 += Pn*forMoon(n, lam, u, i, v)*i

    Mq = s2
    
    return M, K, Pq, Mq

def Yamero(name) :
    inf = 100
    
    Tc = R1
    Ts = R1+G1+B1+R2+G2+B2
    Tw = R1+G1+B1+R2+G2+B2+R3+G3+B3

    Tc = 5
    Ts = 43
    Tw = 71

    lam = 1/Tc
    u = 1/Ts
    p = lam/u

    M = []
    K = []
    Pq = []
    Mq = []
    
    for n in range(1, 21) :
        M_, K_, Pq_, Mq_ = moon(n, lam, u, Tw, inf)

        M.append(M_)
        K.append(K_)
        Pq.append(Pq_)
        Mq.append(Mq_)

    n = diaposon(1, 20, 1)
    plot([M], [n], ['M (операторов)'], ['M (операторов)', 'n'], name+' M (операторов)', '--o')
    plot([K], [n], ['k (операторов)'], ['k (операторов)', 'n'], name+' k (операторов)', '--o')
    plot([Pq], [n], ['P (очереди)'], ['P (очереди)', 'n'], name+' P (очереди)', '--o')
    plot([Mq[0:]], [n[0:]], ['M (очереди)'], ['M (очереди)', 'n'], name+' M (очереди)', '--o')

def Star(n, m, lam, u) :
    s1 = 0
    for i in range(1, m+1) :
        temp = lam**i / u**i / fac(i)
        for j in range(0, i-1+1) : temp *= (n-j)
        s1 += temp

    s2 = 0
    for i in range(m+1, n+1) :
        temp = lam**i / u**i / fac(m) / m**(i-m)
        for j in range(0, i-1+1) : temp *= (n-j)
        s2 += temp

    P0 = 1 / (1 + s1 + s2)
    print(n, m, P0)

    P = [P0, n*(lam/u)*P0]
    for i in range(2, m+1) :
        P.append(P[-1] * (n-i+1) /i * (lam/u))
    for i in range(m+1, n+1) :
        P.append(P[-1] * (n-i+1) /m * (lam/u))

    M = 0
    for i in range(0, n+1) :
        M += i * P[i]
        
    Mwait = 0
    for i in range(m+1, n+1) :
        Mwait+= (i-m) * P[i]

    Pwait = 0
    for i in range(m+1, n+1) :
        Pwait += P[i]

    Mbusy = 0
    s1 = 0
    for i in range(0, m+1) :
        s1 += i*P[i]
    s2 = 0
    for i in range(m+1, n+1) :
        s2 += P[i]
    Mbusy = s1 + m*s2

    Kbusy = Mbusy/m

    return M, Mwait, Pwait, Mbusy, Kbusy

def faith(name) :
    n = G1 + B1 + R2 + B2 + R3 + G3
    Tc = R1 + G1 + B1 + R2 + G2 + B2 + R3
    Ts = R1 + G1 + G2 + B2 + B3 + R3

    lam = 1/Tc
    u = 1/Ts
    p = lam/u

    M = []
    Mwait = []
    Pwait = []
    Mbusy = []
    Kbusy = []
    
    for m in range(1, 41) :
        M_, Mwait_, Pwait_, Mbusy_, Kbusy_ = Star(n, m, lam, u)

        M.append(M_)
        Mwait.append(Mwait_)
        Pwait.append(Pwait_)
        Mbusy.append(Mbusy_)
        Kbusy.append(Kbusy_)

    n = diaposon(1, 40, 1)
    plot([M], [n], ['M (прост)'], ['M (прост)', 'n'], name+' M (прост)', '--o')
    plot([Mwait], [n], ['M (ожид)'], ['M (ожид)', 'n'], name+' M (ожид)', '--o')
    plot([Pwait], [n], ['P (ожид)'], ['P (ожид)', 'n'], name+' P (ожид)', '--o')
    plot([Mbusy], [n], ['M (наладчиков)'], ['M (наладчиков)', 'n'], name+' M (наладчиков)', '--o')
    plot([Kbusy], [n], ['k (наладчиков)'], ['k(наладчиков)', 'n'], name+' k (наладчиков)', '--o')

    
def createNames(s, n) :
    names = []
    for i in n : names.append(s + str(i))
    return names
    

if TASK1_1 :
    t1_1('Задача 1.1')

if TASK1_2 :
    t1_2('Задача 1.2')
    t1_22('Задача 1.2')

if TASK1_3 :
    Miku('Задача 1.3')

if TASK1_4 :
    Yamero('Задача 1.4')

if TASK2 :
    faith('Задача 2.1')







