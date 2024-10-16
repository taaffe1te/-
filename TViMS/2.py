import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy
import math
import os

TASK1_1 = False
TASK1_3 = False
TASK1_4 = False
TASK1_5 = False
TASK1_6 = False
TASK1_7 = False
TASK2 = False
TASK3 = True
TASK4_1 = False
TASK4_2 = False

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
    return
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

def Bernoulli(k, n, p) :
    return C(k, n) * p**k * (1-p)**(n-k)

def Poisson(k, n, p) :
    lam = n*p
    P = lam**k * math.e**(-lam) / fac(k)
    return P

def diaposon(x, y, c) :
    q = [x]
    while (x < y) :
        x += c
        q.append(x)

    return q

def x_1_3(k, n, p) :
    return (k - n*p) / (n*p*(1-p))**0.5

def Pn_1_3(k, n, p) :
    return phi(x_1_3(k, n, p)) / (n*p*(1-p))**0.5

def Pn_1_4(n, p, e) :
    return 2*(scipy.stats.norm.cdf(e / (n*p*(1-p))**0.5)-0.5)

def Pn_1_5(n, p, e) :
    return 2*(scipy.stats.norm.cdf(e*n**0.5 / (p*(1-p))**0.5)-0.5)

def k_1_6(P, n, p) :
    phi = P/2
    x = lapInv(phi, 0.001)
    #print('x2 =', x)
    k = x * (n*p*(1-p))**0.5
    #print('p = ', p)
    #print('k = ', k)

    M = n*p
    return k

def n_1_7(P, tn, p) :
    phi1 = P - 0.5
    x1 = lapInv(phi1, 0.001)

    n = (x1*(tn*p*(1-p))**0.5 + tn) / p

    #print('P =', P, '; phi =', phi1, '; x =', x1, '; n =', n)
    
    return n

def phi(x) :
    return math.e**(-x*x/2) / (2*math.pi)**0.5

def lapInv(c, e) :
    c += 0.5
    y = 0
    x = 5
    q = scipy.stats.norm.cdf((x+y)/2)
    
    while abs(q-c) > e :
        
        if q > c : x = (x+y)/2
        else : y = (x+y)/2
        
        q = scipy.stats.norm.cdf((x+y)/2)
        #print(c, q)

    #print('x =', x)
    return x

def plot(y, x, l, lh, name, lf='', show=True, lw=1.5) :
    plt.figure(figsize=(16, 8))
    for i in range(len(x)) :
        plt.plot(x[i], y[i], lf, alpha=0.7, label=l[i], lw=lw, mec='b', mew=2, ms=10)
    plt.ylabel(lh[0])
    plt.xlabel(lh[1])
    plt.title(name)
    plt.grid()
    #plt.margins(0)
    
    if show :
        plt.legend()
        #save(name=name, fmt='pdf')
        save(name=name, fmt='png')
        plt.show()

def t1_1(n, k, name) :
    p = R1 / (R1+G1+B1)
    k = diaposon(0, k, 1)
    v = []
    for i in k : v.append(Bernoulli(i, n, p))
    plot([v], [k], ['P(k)'], ['P(k)', 'k'], name, '--o')

def t1_2(n, k, name) :
    p = R1 / (R1+G1+B1)
    k = diaposon(0, k, 1)
    v = []
    for i in k : v.append(Bernoulli(i, n, p))
    q = [0]
    k2 = [0]
    for i in range(0, len(v)) :
        k2.append(k[i])
        k2.append(k[i])
        q.append(q[-1])
        q.append(q[-1]+v[i])
    
    plot([q], [k2], 'F(x)', ['F(x)', 'x'], name, '', True, 2.5)

def t1_3(n, k, name) :
    p = R1 / (R1+G1+B1)
    k = diaposon(0, k, 0.1)
    v = []
    for i in k : v.append(Pn_1_3(i, n, p))
    plot([v], [k], ['P(k)'], ['P(k)','k'], name, '', True, 2.5)

def t1_4(name) :
    p = R1 / (R1+G1+B1)
    e = R1
    n = [25, 50, 100, 200, 400]
    v = []
    for i in n : v.append(Pn_1_4(i, p, e))
    plot([v], [n], ['P(|k-M(k)| <= R1))'], ['P(|k-M(k)| <= R1)','n'], name, '-o', True, 2.5)

def t1_5(name) :
    p = R1 / (R1+G1+B1)
    e = [1e-1, 1e-2, 1e-3]
    n = 1000
    v = []
    for i in e : v.append(Pn_1_5(n, p, i))
    plot([v], [e], ['P(k)'], ['P(k)','eps'], name, '-o', True, 2.5)

def t1_6(name) :
    p = R1 / (R1+G1+B1)
    P = [0.7, 0.8, 0.9, 0.95, 0.99]
    n = 1000
    M = n*p
    
    v1 = []
    v2 = []
    
    for i in P :
        k = k_1_6(i, n, p)
        v1.append(M - k)
        v2.append(M + k)

    plot([P, P], [v1, v2] , ['M-k', 'M+k'], ['P','k'], name, '-o', False)
    plt.vlines(M, min(P), max(P), label='M = ' + str(round(M, 3)), lw=3, color='black')
    plt.legend()
    save(name=name, fmt='png')
    plt.show()

def t1_7(name) :
    p = R1 / (R1+G1+B1)
    P = [0.7, 0.8, 0.9, 0.95, 0.99]
    tn = (R1+G1+B1)
    
    v = []
    
    for i in P : v.append(n_1_7(i, tn, p))

    plot([v], [P] , ['P(|k-M(k)| <= R1)'], ['n', 'P(|k-M(k)| <= R1)'], name, '-o', True, 2.5)

def t2(name1, name2) :
    S = (R2+G2+B2)
    n = G2 + B2
    k = diaposon(1, R2, 1) # [i for i in range(1, R2+1)]
    P = []

    M = 0
    M2 = 0
    for i in k :
        P.append(C(i, n) * fac(n)*fac(R2)*fac(S-n) / (fac(i)*fac(R2-i)*fac(S)))
        M += i*P[-1]
        M2 += i**2*P[-1]
    D = M2 - M**2
    
    plot([P], [k], ['P(k)'], ['P(k)', 'k'], name1, '--o', False)
    plt.vlines(M-D, min(P), max(P), label='M - D', lw=2, color='black', linestyle='--')
    plt.vlines(M+D, min(P), max(P), label='M + D', lw=2, color='black', linestyle='--')
    plt.hlines(max(P)*2/3, M, M+D, label='D = ' + str(round(D, 3)), lw=2, color='orange', linestyle='--')
    plt.vlines(M, min(P), max(P), label='M = ' + str(round(M, 3)), lw=3, color='#229F22', linestyle='-.')
    plt.legend()
    save(name=name1, fmt='png')
    plt.show()

    k2 = [0]
    v = [0]
    q = 0
    for i in range(len(k)) :
        k2.append(k[i])
        k2.append(k[i])
        v.append(q)
        q += P[i]
        v.append(q)
    k2.append(k2[-1]+1)
    v.append(v[-1])

    plot([v], [k2], ['F(x)'], ['F(x)', 'x'], name2, '', True, 2.5)

def t3(name1, name2) :
    S = (R3+G3+B3)
    k = diaposon(1, S, 1) # [i for i in range(1, R2+1)]
    P = []

    M = 0
    M2 = 0
    for i in k :
        P.append(C(R3-1, i-1) / C(R3, S))
        M += i*P[-1]
        M2 += i**2*P[-1]
    D = M2 - M**2
    
    plot([P], [k], ['P(k)'], ['P(k)', 'k'], name1, '--o', False)
    plt.vlines(M-D, min(P), max(P), label='M - D', lw=2, color='black', linestyle='--')
    plt.vlines(M+D, min(P), max(P), label='M + D', lw=2, color='black', linestyle='--')
    plt.hlines(max(P)*2/3, M, M+D, label='D = ' + str(round(D, 3)), lw=2, color='orange', linestyle='--')
    plt.vlines(M, min(P), max(P), label='M = ' + str(round(M, 3)), lw=3, color='#229F22', linestyle='-.')
    plt.legend()
    save(name=name1, fmt='png')
    plt.show()

    k2 = [0]
    v = [0]
    q = 0
    for i in range(len(k)) :
        k2.append(k[i])
        k2.append(k[i])
        v.append(q)
        q += P[i]
        v.append(q)
    k2.append(k2[-1]+1)
    v.append(v[-1])

    plot([v], [k2], ['F(x)'], ['F(x)', 'x'], name2, '', True, 2.5)

S = (R1+G1+B1 + R2+G2+B2 + R3+G3+B3 + 1)
p = 1/S

def t4_1(n, k, name, lf='--o') :
    S = (R1+G1+B1 + R2+G2+B2 + R3+G3+B3 + 1)
    p = 1/S
    
    k = diaposon(0, k, 1)
    v = []
    for i in k : v.append(Poisson(i, n, p))
    
    if lf == '--o' : plot([v], [k], ['P(k)'], ['P(k)', 'k'], name, lf)
    else : plot([v], [k], ['P(k)'], ['P(k)', 'k'], name, lf, True, 2.5)

def P4_2(n, p) :
    q = 0
    for i in range(0, 2+1) :
        q += Poisson(i, n, p)
    return 1-q    

def n4_2(P, p) :
    minN = 0
    maxN = 900

    while (minN + 1 != maxN) :
        m = (minN + maxN) // 2

        if P4_2(m, p) <= P : minN = m
        else : maxN = m

        #print('m = ', m, '; P4_2(m, p) =', P4_2(m, p), '; P =', P)

    #raise SystemExit

    return minN

def t4_2(name1, name2) :
    S = (R1+G1+B1 + R2+G2+B2 + R3+G3+B3 + 1)
    p = 1/S

    n = diaposon(1, 800, 1)

    v = []
    for i in n :
        v.append(P4_2(i, p))
    plot([n], [v], ['P(k>=3)'], ['n', 'P(k>=3)'], name1, '', True, 3.5)

    
    P = [0.7, 0.8, 0.9, 0.95, 0.99]
    n = []
    for i in P :
        n.append(n4_2(i, p))
        #print('P =', i, '; n =', n[-1], '; P(n)=', P4_2(n[-1], p))
        
    plot([n], [P], ['P(k>=3)'], ['n', 'P(k>=3)'], name2, '--o')
    
    

if TASK1_1 :
    t1_1(6, 6, 'Задача 1.1 (n=6)')
    t1_1(9, 9, 'Задача 1.1 (n=9)')
    t1_1(12, 12, 'Задача 1.1 (n=12)')
    t1_2(12, 12, 'Задача 1.2')

if TASK1_3 :
    t1_3(25, 19, 'Задача 1.3 (n=25)')
    t1_3(50, 35, 'Задача 1.3 (n=50)')
    t1_3(100, 52, 'Задача 1.3 (n=100)')
    t1_3(200, 100, 'Задача 1.3 (n=200)')
    t1_3(400, 200, 'Задача 1.3 (n=400)')
    t1_3(1000, 500, 'Задача 1.3 (n=1000)')

if TASK1_4 :
    t1_4('Задача 1.4')

if TASK1_5 :
    t1_5('Задача 1.5')

if TASK1_6 :
    t1_6('Задача 1.6')

if TASK1_7 :
    t1_7('Задача 1.7')

if TASK2 :
    t2('Задача 2.1, 2.3, 2.4', 'Задача 2.2')

if TASK3 :
    t3('Задача 3.1, 3.3, 3.4', 'Задача 3.2')

if TASK4_1 :
    t4_1(100, 6, 'Задача 4.1 (n=100)')
    t4_1(1000, 26, 'Задача 4.1 (n=1000)')
    t4_1(10000, 149, 'Задача 4.1 (n=10000)', '-')

if TASK4_2 :
    t4_2('Задача 4.2 (вспомогательный график)', 'Задача 4.2')







