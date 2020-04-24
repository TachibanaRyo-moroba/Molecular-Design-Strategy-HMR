#!/usr/bin/env python
# coding: utf-8


import os
import traceback
import shutil
import time
import datetime
import random
import itertools
import subprocess
import math
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



def HMR_Identify(mol_data, file_name, preCon=[]):
    mol = mol_data.loc[:, ["atomTYPE", "x", "y", "z", "charge"]].copy().reset_index()
    mol["id"] = np.array([i for i in range(len(mol))]) ; mol["atomID"] = mol["id"] + 1 
    mol = mol.loc[:, ["id", "atomID", "atomTYPE", "x", "y", "z", "charge"]]
    mol = mol.astype({"id":int, "atomID":int, "atomTYPE":str, "x":float, "y":float, "z":float, "charge":float})
    
    if "open" in file_name.lower():
        form = "open"
    elif "close" in file_name.lower():
        form = "close"
    else:
        form = "open"
        print("File name does not include \"open\" nor \"closed\". This may not be HMR derivative.")
        
    
    dis = np.zeros((len(mol),len(mol)), dtype=float)  
    for i in range(len(mol)):
        for j in range(len(mol)):
            dis[i,j] = np.linalg.norm(mol.loc[i,["x", "y", "z"]].values - mol.loc[j,["x", "y", "z"]].values)

    if len(preCon)>0:
        con = np.array(np.array(preCon>0.6,dtype=bool),dtype=int).copy()
        for i in range(len(mol)):
            if mol.loc[i,"atomTYPE"]=="H" and con[i,:].sum()==0:
                for j in range(len(mol)):
                    if mol.loc[j,"atomTYPE"] in ["Si","P","S"]:
                        con[i,j] = dis[i,j]<1.5
                        con[j,i] = con[i,j]
                    elif mol.loc[j,"atomTYPE"] in ["C","N","O"]:
                        con[i,j] = dis[i,j]<1.2
                        con[j,i] = con[i,j]
    else:
        con = np.zeros((len(mol),len(mol)), dtype=int)
        for i in range(len(mol)):
            for j in range(len(mol)):
                atmPair = mol.loc[np.array([i,j]),"atomTYPE"].values.tolist()
                if i == j:
                    con[i,j] = 0
                elif "H" in atmPair:
                    if ("Si" in atmPair)or("P" in atmPair)or("S" in atmPair):
                        con[i,j] = dis[i,j]<1.5
                    elif atmPair.count("H")==2:
                        con[i,j] = 0
                    else:
                        con[i,j] = dis[i,j]<1.2
                elif ("Si" in atmPair)or("P" in atmPair)or("S" in atmPair)or("Cl" in atmPair):
                    con[i,j] = dis[i,j]<2.1
                else:
                    con[i,j] = dis[i,j]<1.6

    mol["bond"] = con.sum(axis=1).astype(int)
        
    #check excessiv bond number
    for ak in ["H", "F", "Cl"]:
        if ak in mol["atomTYPE"].values:
            if mol.loc[mol["atomTYPE"]==ak,"bond"].max()>1:
                print(mol)
                raise ValueError(f"Some {ak} atoms have wrong bonds.")
    for ak in ["C", "O", "N", "Si"]:
        if ak in mol["atomTYPE"].values:
            if mol.loc[mol["atomTYPE"]==ak,"bond"].max()>5:
                print(mol)
                raise ValueError(f"Some {ak} atoms have wrong bonds.")
     

    #find position 9 in xanthene
    pos9_cand = []
    for ci in mol[mol["atomTYPE"]=="C"]["id"].values.astype(np.int64).tolist():
        link_a = np.where(con[ci,:])[0]
        link_k = []
        for a in link_a:
            link_k.append(str(mol.loc[a,"bond"]) + mol.loc[a,"atomTYPE"])
        if link_k.count("3C") == 3:
            pos9_cand.append(ci)
    
    c_o = ["close", "open"]
    while True:
        mol["pos"] = ["None" for i in range(len(mol))]
        mol["d_from9"] = np.full(len(mol),-1, dtype=int)
        c_o.remove(form)
        if form == "close":        
            for can in pos9_cand:
                if mol.loc[can,"bond"] == 4:
                    mol.loc[can,"pos"] = "xant9"
                    mol.loc[can,"d_from9"] = 0
        elif form == "open":
            for can in pos9_cand:
                link_a = np.where(con[can,:])[0]
                link_c = []
                for a in link_a:
                    if a in pos9_cand:
                        link_c.append(1)
                    else:
                        link_c.append(0)
                if link_c.count(0)<=1:
                    mol.loc[can,"pos"] = "xant9"
                    mol.loc[can,"d_from9"] = 0

        if len(mol.loc[mol["pos"]=="xant9", "id"].values)==1:
            break
        elif len(c_o)==0:
            raise ValueError("This structure is not close, nor open.")
        else:
            form = c_o[0]

    d = 1
    flag = True
    while (flag):
        flag = False
        for at in mol.loc[mol["d_from9"]==d-1, "id"].values.tolist():
            link_a = np.where(con[at,:])[0]
            for a in link_a:
                if mol.loc[a,"d_from9"] == -1:
                    flag = True
                    mol.loc[a,"d_from9"] = d
        d += 1

    #find position Xsc
    mol["d_fromXsc"] = np.full(len(mol),-1, dtype=int)
    if form == "close": 
        ai = mol.loc[(mol["d_from9"]==1)&(~(mol["atomTYPE"]=="C")),"id"].iloc[0]
        mol.loc[ai,"pos"]="Xsc"
        mol.loc[ai,"d_fromXsc"] = 0
    elif form == "open":
        xant9pos = mol.loc[mol["pos"]=="xant9", ["x","y","z"]].iloc[0].values.astype(np.float64)
        p1 = mol.loc[mol["d_from9"]==1, ["x","y","z"]].iloc[0].values.astype(np.float64)
        p2 = mol.loc[mol["d_from9"]==1, ["x","y","z"]].iloc[1].values.astype(np.float64)
        vec = np.cross(p1-xant9pos, p2-xant9pos) ; vec /= np.linalg.norm(vec)
        L_ai = mol.loc[(mol["d_from9"]==4)&((mol["atomTYPE"]=="O")|(mol["atomTYPE"]=="N")),"id"]
        highest_inn = 0. 
        for ai in L_ai:
            link_a = np.where(con[ai,:])[0]
            for a in link_a:
                atp = str(mol.loc[a,"bond"]) + mol.loc[a,"atomTYPE"]
                if atp == "4C" and mol.loc[a,"d_from9"]==3:
                    cpos = mol.loc[a, ["x","y","z"]].values.astype(np.float64)
                    cvec = (cpos-xant9pos)/np.linalg.norm(cpos-xant9pos)
                    if np.abs(np.inner(cvec, vec)) > highest_inn:
                        highest_inn = np.abs(np.inner(cvec, vec))
                        xscid = ai
        mol.loc[xscid,"pos"]="Xsc"
        mol.loc[xscid,"d_fromXsc"] = 0
        
    d = 1
    flag = True
    while (flag):
        flag = False
        for at in mol.loc[mol["d_fromXsc"]==d-1, "id"].values.tolist():
            link_a = np.where(con[at,:])[0]
            for a in link_a:
                if mol.loc[a,"d_fromXsc"]==-1:
                    flag = True
                    mol.loc[a,"d_fromXsc"]=d
        d += 1

    #find position 10 in xanthene  
    mol["d_from10"] = np.full(len(mol),-1, dtype=int)
    if form == "close":
        L_ai = mol.loc[(mol["d_fromXsc"]==4)&(mol["d_from9"]==3)&                       (((mol["atomTYPE"]=="O")&(mol["bond"]==2))|((~(mol["atomTYPE"]=="H"))&(mol["bond"]==4))),"id"]
    elif form == "open":
        L_ai = mol.loc[(mol["d_fromXsc"]==7)&(mol["d_from9"]==3)&                       (((mol["atomTYPE"]=="O")&(mol["bond"]==2))|((~(mol["atomTYPE"]=="H"))&(mol["bond"]==4))),"id"]
    for ai in L_ai:
        link_a = np.where(con[ai,:])[0]
        link_k = []
        for a in link_a:
            link_k.append(str(mol.loc[a,"bond"]) + mol.loc[a,"atomTYPE"])
        if (link_k.count("3C") == 2)and(link_k.count("1H") == 0):
            mol.loc[ai,"pos"] = "xant10"
            mol.loc[ai,"d_from10"] = 0

    d = 1
    flag = True
    while (flag):
        flag = False
        for at in mol.loc[mol["d_from10"]==d-1, "id"].values.tolist():
            link_a = np.where(con[at,:])[0]
            for a in link_a:
                if mol.loc[a,"d_from10"]==-1:
                    flag = True
                    mol.loc[a,"d_from10"]=d
        d += 1

    #find position Xani
    mol.loc[(mol["d_from9"]==5)&(mol["d_from10"]==4)&((mol["atomTYPE"]=="N")|(mol["atomTYPE"]=="O")),"pos"] = "Xani"
    xani = mol.loc[mol["pos"]=="Xani", "atomTYPE"].values.tolist()
    if (xani.count("N")==2)and(xani.count("O")==0):
        fluo = "Rhodamine"
    elif (xani.count("N")==1)and(xani.count("O")==1):
        fluo = "OH-Rhodol"
        Oani = mol.loc[(mol["pos"]=="Xani")&(mol["atomTYPE"]=="O"),"id"].iloc[0]
        link_a = np.where(con[Oani,:])[0]
        for a in link_a:
            if mol.loc[a,"atomTYPE"]=="C" and mol.loc[a,"bond"]==4:
                fluo = "OR-Rhodol"    
    elif (xani.count("N")==0)and(xani.count("O")==2):
        fluo = "Fluorescein"
    else:
        fluo = "Unknown"
    #find position position 1,8 in xanthene 
    mol.loc[(mol["d_from9"]==2)&(mol["d_from10"]==3)&(mol["atomTYPE"]=="C")&(mol["bond"]==3),"pos"] = "xant1&8"
    #find position position 2,7 in xanthene 
    mol.loc[(mol["d_from9"]==3)&(mol["d_from10"]==4)&(mol["atomTYPE"]=="C")&(mol["bond"]==3),"pos"] = "xant2&7"
    #find position position 3,6 in xanthene 
    mol.loc[(mol["d_from9"]==4)&(mol["d_from10"]==3)&(mol["atomTYPE"]=="C")&(mol["bond"]==3),"pos"] = "xant3&6"
    #find position position 4,5 in xanthene 
    mol.loc[(mol["d_from9"]==3)&(mol["d_from10"]==2)&(mol["atomTYPE"]=="C")&(mol["bond"]==3),"pos"] = "xant4&5"

    #find position position in benzene
    mol.loc[(mol["d_from9"]==1)&(mol["d_from10"]==4)&(mol["atomTYPE"]=="C")&(mol["bond"]==3),"pos"] = "ben1"
    mol.loc[(mol["d_from9"]==2)&(mol["d_from10"]==5)&(mol["d_fromXsc"]==2)&(mol["atomTYPE"]=="C")&(mol["bond"]==3),"pos"] = "ben2"
    mol.loc[(mol["d_from9"]==3)&(mol["d_from10"]==6)&(mol["d_fromXsc"]==3)&(~(mol["atomTYPE"]=="H")),"pos"] = "ben3"
    mol["d_fromB3"] = np.full(len(mol),-1, dtype=int)
    mol.loc[(mol["d_from9"]==3)&(mol["d_from10"]==6)&(mol["d_fromXsc"]==3)&(~(mol["atomTYPE"]=="H")),"d_fromB3"] = 0
    d = 1
    flag = True
    while (flag):
        flag = False
        for at in mol.loc[mol["d_fromB3"]==d-1, "id"].values.tolist():
            link_a = np.where(con[at,:])[0]
            for a in link_a:
                if mol.loc[a,"d_fromB3"] == -1:
                    flag = True
                    mol.loc[a,"d_fromB3"] = d
        d += 1

    if len(mol.loc[(mol["d_from9"]==2)&(mol["d_fromB3"]==2)&(mol["d_fromXsc"]>=2)])==0:
        benzene = 6
    else:
        benzene = 5

    if benzene == 6:
        mol.loc[(mol["d_fromXsc"]>2)&(mol["d_from9"]==4)&(mol["d_from10"]==7)&                (((mol["atomTYPE"]=="C")&(mol["bond"]==3))|((mol["atomTYPE"]=="N")&(mol["bond"]==2))),"pos"] = "ben4"
        mol.loc[(mol["d_fromXsc"]>2)&(mol["d_from9"]==3)&(mol["d_from10"]==6)&(mol["d_fromB3"]==2)&                (((mol["atomTYPE"]=="C")&(mol["bond"]==3))|((mol["atomTYPE"]=="N")&(mol["bond"]==2))),"pos"] = "ben5"
        mol.loc[(mol["d_fromXsc"]>2)&(mol["d_from9"]==2)&(mol["d_from10"]==5)&(mol["d_fromB3"]==3)&                (((mol["atomTYPE"]=="C")&(mol["bond"]==3))|((mol["atomTYPE"]=="N")&(mol["bond"]==2))),"pos"] = "ben6"
    elif benzene == 5:
        mol.loc[(mol["d_fromXsc"]>2)&(mol["d_from9"]==3)&(mol["d_from10"]==6)&(mol["d_fromB3"]==1)&(mol["bond"]>1)&                (~(mol["atomTYPE"]=="H")),"pos"] = "ben4"
        mol.loc[(mol["d_fromXsc"]>2)&(mol["d_from9"]==2)&(mol["d_from10"]==5)&(mol["d_fromB3"]==2)&(mol["bond"]>1)&                (~(mol["atomTYPE"]=="H")),"pos"] = "ben5"

    #find position othoer positions 
    mol.loc[(mol["d_fromXsc"]==1)&(mol["d_from9"]>1)&(~(mol["atomTYPE"]=="H"))&(mol["d_fromB3"]==2),"pos"] = "HMC"    
    mol.loc[(mol["pos"]=="Xani")&(mol["bond"]==4)&(mol["atomTYPE"]=="N"),"pos"] = "catXani"
    mol.loc[(mol["pos"]=="Xani")&(mol["bond"]==3)&(mol["atomTYPE"]=="O"),"pos"] = "catXani"
    mol.loc[(mol["pos"]=="None")&(mol["d_from9"]==1),"pos"] = "xant"
    mol.loc[(mol["pos"]=="None")&(mol["d_from9"]==2)&(mol["d_from10"]==1),"pos"] = "xant"
    mol.loc[mol["d_from9"]==-1, "pos"] = "water"
    
    for HMRpos in ["ben"+str(n) for n in range(1, benzene+1)]                        +["xant"+str(n)+"&"+str(9-n) for n in range(1,5)]+["xant10","Xsc","Xani","catXani","HMC"]:
        for posid in  mol.loc[mol["pos"]==HMRpos, "id"].values:
            cand = np.where(con[posid,:])[0].tolist()
            while cand:
                ci = cand.pop(0)
                if mol.loc[ci, "pos"]=="None":
                    mol.loc[ci, "pos"] = "group-"+HMRpos
                    cand += np.where(con[ci,:])[0].tolist()
    
    #check identifitaion
    err = False ; mess = ""
    for p in ["xant1&8","xant2&7","xant3&6","xant4&5"]:
        if not len(mol.loc[mol["pos"]==p, "id"].values)==2:
            err = True ; mess += "xant1-8 is wrong"         
    for p in ["ben"+str(i) for i in range(1, benzene+1)]+["xant9","xant10","Xsc","HMC"]:
        if not len(mol.loc[mol["pos"]==p, "id"].values)==1:
            err = True ; mess += ", xant9-10,Xsc or HMC is wrong"            
    if not len(mol.loc[(mol["pos"]=="Xani")|(mol["pos"]=="catXani"), "id"].values)==2:
        err = True ; mess += ", Xani is wrong."
    if len(mol.loc[(mol["pos"]=="None"), "id"].values)>0:
        err = True ; mess += ", \"None\" exists."
        
    if err:
        print(file_name+" Identifitaion may be invalid.\n")
        print(mess)
        print(mol)
    
    #identify Bond formation 
    conB = np.array(con, dtype=float)
    bnd = np.where(con) 
    benP = ["ben"+str(i) for i in range(1, benzene+1)] ; xantP = ["xant9","xant1&8","xant2&7","xant3&6","xant4&5", "xant"]
    sp2 = ["3C", "2N", "1O"] ; sp3 = ["2C", "1N"]
    for i, j in zip(bnd[0],bnd[1]):
        pi = str(mol.loc[i,"pos"]) ; ki = str(mol.loc[i,"bond"]) + mol.loc[i,"atomTYPE"]
        pj = str(mol.loc[j,"pos"]) ; kj = str(mol.loc[j,"bond"]) + mol.loc[j,"atomTYPE"]
        if (pi in benP)and(pj in benP):
            conB[i,j] = 1.5
        elif (pi in xantP)and(pj in xantP):
            conB[i,j] = 1.5
        elif (pi in ["ben1","xant9"])and(pj in ["ben1","xant9"]):
            conB[i,j] = 1
        elif (ki in sp2)and(kj in sp2):
            conB[i,j] = 2
        elif(ki in sp3)and(kj in sp3):
            conB[i,j] = 3
    
    #find HBond formation    
    WATERS = mol.loc[(mol["pos"]=="water")&(mol["atomTYPE"]=="O")]
    water_num = len(WATERS)
    
    HBcand = mol.loc[(mol["atomTYPE"]=="O")|(mol["atomTYPE"]=="N")|(mol["atomTYPE"]=="S")]
    mol["HBond"] = "_"
    Xanis = mol.loc[(mol["pos"]=="Xani")|(mol["pos"]=="catXani"),["pos", "id"]].values.tolist()
    factors = ["Xsc_"+str(mol.loc[mol["pos"]=="Xsc","id"].iloc[0]), 
               "xant10_"+str(mol.loc[mol["pos"]=="xant10","id"].iloc[0]),
               Xanis[0][0]+"_"+str(Xanis[0][1]),
               Xanis[1][0]+"_"+str(Xanis[1][1])]
    for hbc in HBcand.loc[~((HBcand["pos"]=="Xsc")|(HBcand["pos"]=="xant10")|                          (HBcand["pos"]=="Xani")|(HBcand["pos"]=="catXani")|(HBcand["pos"]=="water")), ["pos","id"]].values:
        factors.append(hbc[0] + "_" + str(hbc[1]))
    for eau in WATERS["id"].values:
        factors.append("water_" + str(eau))
        
    Hbon = pd.DataFrame(np.zeros((len(factors),len(factors)), dtype=int), columns=factors, index=factors)
    for f1 in factors:
        f1_id = int(f1.split("_")[1])
        f1_pos = f1.split("_")[0]
        for f2 in factors:
            f2_id = int(f2.split("_")[1])
            f2_pos = f2.split("_")[0]
            if (not(f1_id==f2_id))and(dis[f1_id,f2_id] < 3.4)and((f1_pos=="water")or(f2_pos=="water")):
                Hbon.loc[f1,f2] = 1 ; Hbon.loc[f2,f1] = 1
                if not str(f1_id) in mol.loc[f2_id, "HBond"]:
                    mol.loc[f2_id, "HBond"] = mol.loc[f2_id, "HBond"] + str(f1_id) + "_"
                if not str(f2_id) in mol.loc[f1_id, "HBond"]:
                    mol.loc[f1_id, "HBond"] = mol.loc[f1_id, "HBond"] + str(f2_id) + "_"

    
    other_data = {}
    other_data["form"] = form
    other_data["benzene"] = benzene
    other_data["water"] = water_num
    other_data["kind"] = fluo
    
    
    return mol, dis, conB, Hbon, other_data




def Read_GaussianOutput(file_path, gf_col=True, moc_col=True, freq_col=True, printLog=False):
    
    if not ((".log" in file_path.lower())or(".out" in file_path.lower())):
        print("This is not Gaussian output file.")
        return 0, 0, 0, 0, 0, 0, 0
    
    start = time.time()
    otherDATAs = {}
    UMOs = [] ; OMOs = []
    Mcharge = []
    pop = True ; opt = True ; freq = True ; gfprint = True 
    normalT = 0 ; failure = False ; TorError = False ; fullPath = False
    afterO = False
    keyWords = [("method","#", "-------",1),
                ("atoms", "Multiplicity =", "GradGradGradGrad",1),
                ("optPath", "Standard orientation:", "Rotational constants",1),
                ("gfprint", "AO basis set (Overlap normalization)", "nuclear repulsion energy",1),   
                ## After Opt
                ("orbitals", "Population analysis using the SCF density", "Molecular Orbital Coefficients:",0),  
                ("MOC", "Molecular Orbital Coefficients:", "Density Matrix",0), 
                ("atomcharge", "Mulliken atomic charges:", "Sum of Mulliken",0),  
                ("atomcharge", "Mulliken charges:", "Sum of Mulliken",0),
                ("otherData", "\\", "Normal termination",0), 
                ("otherData", "|", "Normal termination",0), 
                ("freq", "and normal coordinates", "GradGradGradGrad",0)]
    
    state = [keyWords[i][0] for i in range(len(keyWords))]
    stFlags = [keyWords[i][1] for i in range(len(keyWords))]
    enFlags = [keyWords[i][2] for i in range(len(keyWords))]
    beforeOPT = [keyWords[i][3] for i in range(len(keyWords))]
    collect = "None"
    
    col_data = {}
    meth = ""
    for i in set(state):
        col_data[i] = []
    col_data["enePath"] = []
    
    with open(file_path, 'r') as output:
        for ln, line in enumerate(output):
            ##Special keywords
            if "Normal termination" in line:
                if printLog: print("Normal termination  " + str(ln+1))
                normalT += 1 
            elif ("%chk" in line) and (":" in line):
                fullPath = True
            elif "Error termination" in line:
                failure = True
            elif ("Tors failed for" in line) or ("Linear angle in Tors" in line):
                TorError = True
            elif "Stationary point found" in line:
                if printLog: print("Stationary Point  " + str(ln+1))
                afterO = True
            elif "Entering Link 1" in line:
                otherDATAs["JobID"] = int(line.strip().split()[-1][:-1])
            elif "SCF Done:" in line:
                lines = line.strip().split()
                col_data["enePath"].append(float(lines[4]))
            ##Set Collect State           
            if not (collect == "None"):
                if enFlags[state.index(collect)] in line:
                    if printLog: print(collect + " to " + str(ln+1))
                    collect = "None"            
            for fi, sf in enumerate(stFlags):
                if (sf in line) and ((len(col_data[state[fi]])==0)or(state[fi] == "optPath")) and (beforeOPT[fi] or afterO ):
                    collect = state[fi]
                    if printLog: print(collect + " from " + str(ln+1))        
                    
            if (enFlags[0] in line)and(col_data["method"])and(not meth):
                meth = "".join(col_data["method"])
                otherDATAs["rootSection"] = meth
                if "scrf" in meth.lower():
                    if "smd" in meth:
                        otherDATAs["solvent"] = "SMD"
                    else:
                        otherDATAs["solvent"] = "PCM"
                else:
                    otherDATAs["solvent"] = "GAS"
                if "ts" in meth.lower():
                    otherDATAs["state"] = "TS"
                else:
                    otherDATAs["state"] = "local"
                if "nosymm" in meth.lower():
                    stFlags[stFlags.index("Standard orientation:")] = "Input orientation:" 
                if not "pop" in meth.lower():
                    pop = False
                    enFlags[enFlags.index("Molecular Orbital Coefficients:")] = "Condensed to atoms"                   
                if not"opt" in meth.lower():
                    opt = False ; afterO = True 
                if not"freq" in meth.lower():
                    freq = False
                if not"gfprint" in meth.lower():
                    gfprint = False                    
                if opt or freq:
                    otherDATAs["Opt_Structure"] = 1
                else:
                    otherDATAs["Opt_Structure"] = 0
                
            
            ##Collect Data
            if collect == "method":
                col_data["method"].append(line.strip().lower())           
            elif not (collect == "None"):
                col_data[collect].append(line.strip())                                
           
                               
    
    molDATA = pd.DataFrame(index=[], columns=["id", "atomID", "atomTYPE", "x", "y", "z"])
    atms = [] 
    for atm in col_data["atoms"]:
        if "Charge" in atm:
            otherDATAs["netCharge"] = int(atm.split()[2])
            otherDATAs["spin"] = int(atm.split()[5])
        at = atm.split()
        if (len(at)==4) and (at[0] in ["C", "H", "N", "O", "S", "Si", "P", "F", "Cl"]): 
            atms.append(at[0])
    molDATA = pd.DataFrame({
        "id":np.array([i for i in range(len(atms))], dtype=int),
        "atomID":np.array([i for i in range(1,len(atms)+1)], dtype=int),
        "atomTYPE":atms
    })
    
    ##"optPath"
    coos = []
    coo = []
    collect = False
    for opath in col_data["optPath"]:
        if ("Standard orientation:" in opath)or("Input orientation:" in opath):
            collect = True
            if coo:
                coo = np.array(coo)
                coos.append(coo)
                coo = []
        elif ("Rotational constants" in opath)or("Distance matrix (angstroms):" in opath):
            collect = False
            
        elif collect :
            opat = opath.split()    
            if len(opat)==6:
                try:
                    aid = int(opat[0])
                    coo.append(opat[3:])
                except:
                    pass            
    coo = np.array(coo, dtype=float)        
    coos.append(coo) 

    if coo.size > 0:
        molDATA["x"] = np.array(coo[:,0], dtype=float)
        molDATA["y"] = np.array(coo[:,1], dtype=float)
        molDATA["z"] = np.array(coo[:,2], dtype=float)
        Z = {"H":1, "C":6, "N":7, "O":8, "F":9, "Si":14, "P":15, "S":16, "Cl":17}
        N_rep = 0
        for i in molDATA["id"].values:
            i_Z = Z[molDATA.loc[i, "atomTYPE"]]
            i_pos = molDATA.loc[i, ["x","y","z"]].values.astype(np.float64)
            for ii in range(i):
                ii_Z = Z[molDATA.loc[ii, "atomTYPE"]]
                ii_pos = molDATA.loc[ii, ["x","y","z"]].values.astype(np.float64)
                R_i_ii = np.linalg.norm(i_pos-ii_pos)/0.52917721
                N_rep += (i_Z*ii_Z)/R_i_ii
        otherDATAs["Nuc_rep"] = N_rep
    
    pathDATA = []
    en = len(col_data["enePath"])
    if en:
        if len(coos) > en:
            for i in range(len(coos)-en):
                col_data["enePath"].append(col_data["enePath"][-1])

        for i in range(len(coos)):
            pathDATA.append((col_data["enePath"][i], coos[i]))
        otherDATAs["E(NoZeroEne)"] = col_data["enePath"][-1]
    
    df_gp = pd.DataFrame(index=[], columns=["atomID", "atomTYPE", "AO_SHAPE", "AO_IDst", "AO_IDen", "Sc1", "Sc2", "Sc3",
                                                                                                         "alpha", "d_s", "d_p"])
    df_moc = pd.DataFrame(index=[], columns=["AO_ID", "atomID", "atomTYPE", "AO_TYPE", "Value"])
    df_freq = pd.DataFrame(index=[], columns=["atomID", "xyz", "Value"])
    
    if fullPath and len(col_data["method"])==0:
        if printLog: print("This calculation ended chk-full-path error.")
        return -4, molDATA, pathDATA, df_gp, df_moc, df_freq, otherDATAs
    elif len(pathDATA)==0 and failure:
        if printLog: print("This calculation ended syntax error.")
        return -3, molDATA, pathDATA, df_gp, df_moc, df_freq, otherDATAs
    elif TorError:
        if printLog: print("This calculation ended coodinate error.")
        return -2, molDATA, pathDATA, df_gp, df_moc, df_freq, otherDATAs
    elif failure:
        if printLog: print("This calculation ended unsuccessfully.")
        return -1, molDATA, pathDATA, df_gp, df_moc, df_freq, otherDATAs
    elif ((not freq) and (normalT<1))or(freq and (normalT<2)):
        if printLog: print("This calculation is yet to be finished.")
        return 0, molDATA, pathDATA, df_gp, df_moc, df_freq, otherDATAs
    
    if printLog:
        print("Read finished.")
        print("elapsed_time:{0}".format(time.time() - start) + "[sec]")

    ## "orbitals"
    for obt in col_data["orbitals"]:
        if ". eigenvalues"in obt:
            ob = obt.split()
            for o in ob[4:]:
                try:
                    if "occ" in ob[1]:
                        OMOs.append(float(o))
                    elif "virt" in ob[1]:
                        UMOs.append(float(o))
                except:
                    for os in o.split("-"):
                        if os:
                            if "occ" in ob[1]:
                                OMOs.append(-1*float(os))
                            elif "virt" in ob[1]:
                                UMOs.append(-1*float(os))
                    
    UMOs = np.sort(np.array(UMOs)) ; OMOs = np.sort(np.array(OMOs))[::-1]
    #otherDATAs["UMOs"] = UMOs[:10] ; otherDATAs["OMOs"] = OMOs[:10]
    otherDATAs["LUMO"] = UMOs.min() ; otherDATAs["HOMO"] = OMOs.max()

    if printLog:
        print("Orb finished.")
        print("elapsed_time:{0}".format(time.time() - start) + "[sec]")
    
    ## "otherData"
    res = "".join(col_data["otherData"])
    if "|" in res:
        results = res.split("|")
    else:
        results = res.split("\\")
    otherDATAs["method"] = results[4]
    otherDATAs["basisSet"] = results[5]


    ## "Atomcharge"
    for cha in col_data["atomcharge"]:
        ch = cha.split()
        if len(ch)>2:
            try:
                Mcharge.append(float(ch[2]))
            except:
                pass
    molDATA["charge"] = np.array(Mcharge, dtype=float)
    
    if printLog:
        print("Coo., charge finished.")
        print("elapsed_time:{0}".format(time.time() - start) + "[sec]")

    ## "gprint" 
    o_num = 0  
    if gfprint and gf_col:
        for g in col_data["gfprint"]:
            gs = g.split()
            if gs[0] == "Atom":
                try:
                    a_id = int(gs[1][1:])
                    a_type = gs[1][0]
                except:
                    a_id = int(gs[1][2:])
                    a_type = gs[1][:2]
                o_shape = gs[4]
                st_id = int(gs[7]) ; en_id = int(gs[9])
                o_num = int(gs[5])
                sc = np.array(gs[10:], dtype=float)
            elif o_num :
                if len(gs)==2:
                    record = pd.Series([a_id, a_type, o_shape, st_id, en_id, sc[0], sc[1], sc[2],
                                float(gs[0].replace("D","E")), float(gs[1].replace("D","E")), 0.], index=df_gp.columns)
                    df_gp = df_gp.append(record, ignore_index=True)
                elif len(gs)==3:
                    record = pd.Series([a_id, a_type, o_shape, st_id, en_id, sc[0], sc[1], sc[2], 
                                float(gs[0].replace("D","E")), float(gs[1].replace("D","E")), 
                                                                        float(gs[2].replace("D","E"))], index=df_gp.columns)
                    df_gp = df_gp.append(record, ignore_index=True)
                o_num -= 1

    if printLog:
        print("GPrint finished.")
        print("elapsed_time:{0}".format(time.time() - start) + "[sec]")
    
    ## "MOC"
    if pop and moc_col:
        record = pd.Series([-1, -1, "None", "None", "Occ."], index=df_moc.columns)
        df_moc = df_moc.append(record, ignore_index=True)
        record = pd.Series([-1, -1, "None", "None", "Irrep."], index=df_moc.columns)
        df_moc = df_moc.append(record, ignore_index=True)
        record = pd.Series([-1, -1, "None", "None", "Eigenvalue"], index=df_moc.columns)
        df_moc = df_moc.append(record, ignore_index=True)

        for m in col_data["MOC"]:
            if "6         7         8         9        10" in m:
                break
            ms = m.split()   
            if (not("--" in m))and("1S" in m):
                ao_id = int(ms[0])
                atm_id = int(ms[1])
                atm_type = ms[2]
                ao_type = ms[3]
                record = pd.Series([ao_id, atm_id, atm_type, ao_type, "Coeff."], index=df_moc.columns)
                df_moc = df_moc.append(record, ignore_index=True)
            elif (not("--" in m))and(len(ms)>6):  
                ao_id = int(ms[0])
                ao_type = ms[1]
                record = pd.Series([ao_id, atm_id, atm_type, ao_type, "Coeff."], index=df_moc.columns)
                df_moc = df_moc.append(record, ignore_index=True)

        for m in col_data["MOC"]:
            if "Molecular Orbital Coefficients" in m:
                continue
            ms = m.split()
            if (not("." in m))and(len(ms)<6): 
                try:                                ## "1         2         3         4         5"
                    for i in ms:
                        df_moc[str(int(i))] = 0.
                    mo_id = ms
                except:                            ## "(A1)--O   (A1)--O   (B2)--O   (A1)--O   (B1)--O 
                    if "--" in m:
                        for mid, i in zip(mo_id, ms):
                            df_moc.loc[df_moc["Value"]=="Irrep.", mid] = i.split("--")[0] 
                            df_moc.loc[df_moc["Value"]=="Occ.", mid] = i.split("--")[1] 
                    else:
                        for mid, i in zip(mo_id, ms):
                            df_moc.loc[df_moc["Value"]=="Irrep.", mid] = "A1"
                            df_moc.loc[df_moc["Value"]=="Occ.", mid] = i 

            elif "Eigenvalues --" in m:        ## "Eigenvalues --   -20.58070  -1.35154  -0.72784  -0.57336  -0.50757"
                if len(mo_id)==len(ms[2:]):
                    for mid, i  in zip(mo_id, ms[2:]):
                        df_moc.loc[df_moc["Value"]=="Eigenvalue", mid] = float(i)
                else:
                    vals = []
                    for msc in ms[2:]:
                        try:
                            vals.append(float(msc))
                        except:
                            for mscs in msc.split("-"):
                                if mscs:
                                    vals.append(-1*float(mscs))
                                    
                    for mid, i  in zip(mo_id, vals):
                        df_moc.loc[df_moc["Value"]=="Eigenvalue", mid] = i
                        
            elif "." in m:
                if "1S" in m:
                    ao_id = int(ms[0])
                    atm_id = int(ms[1])
                    atm_type = ms[2]
                    ao_type = ms[3]
                    df_moc.loc[(df_moc["AO_ID"]==ao_id)&(df_moc["atomID"]==atm_id)&                               (df_moc["atomTYPE"]==atm_type)&(df_moc["AO_TYPE"]==ao_type),mo_id] = np.array(ms[4:], dtype=float)        
                else:
                    ao_id = int(ms[0])
                    ao_type = ms[1]
                    df_moc.loc[(df_moc["AO_ID"]==ao_id)&(df_moc["atomID"]==atm_id)&(df_moc["atomTYPE"]==atm_type)&                                (df_moc["AO_TYPE"]==ao_type),mo_id] = np.array(ms[-1*len(mo_id):], dtype=float)


    if printLog:
        print("MOC finished.")
        print("elapsed_time:{0}".format(time.time() - start) + "[sec]")
    
    ## "freq"        
    if freq:         
        S_collect = False
        for f in col_data["freq"]:
            if "Zero-point correction=" in f:
                otherDATAs["Zero_Ene"] = float(f.strip().split()[2]) 
            elif "Sum of electronic and thermal Free Energies" in f:
                otherDATAs["G"] = float(f.strip().split()[7])
            elif "Sum of electronic and thermal Enthalpies" in f:
                otherDATAs["H"] = float(f.strip().split()[6])
            elif "Sum of electronic and thermal Energies" in f:
                otherDATAs["E"] = float(f.strip().split()[6])
            elif ("Temperature" in f)and("Pressure" in f):
                otherDATAs["Temperature(K)"] = float(f.strip().split()[1])
                otherDATAs["Pressure"] = float(f.strip().split()[4])
                otherDATAs["E_rot(kJ/mol)"] = 1.5*8.31446*otherDATAs["Temperature(K)"]/1000.
                otherDATAs["E_trans(kJ/mol)"] = otherDATAs["E_rot(kJ/mol)"]
            elif "Molecular mass" in f:
                otherDATAs["Mol_Mass"] = float(f.strip().split()[2])
            elif "E (Thermal)" in f:
                S_collect = True
            elif "Log10(Q)" in f:
                S_collect = False
            elif S_collect and ("Translational" in f):
                otherDATAs["S_trans(kJ/mol-K)"] = float(f.strip().split()[3])*4.184/1000.
            elif S_collect and ("Rotational" in f):
                otherDATAs["S_rot(kJ/mol-K)"] = float(f.strip().split()[3])*4.184/1000.
     
            
    if freq and freq_col:
        df_freq = df_freq.append(pd.Series([-1, "None", "Freq."], index=df_freq.columns), ignore_index=True)     
        df_freq = df_freq.append(pd.Series([-1, "None", "Mass"], index=df_freq.columns), ignore_index=True)
        df_freq = df_freq.append(pd.Series([-1, "None", "Force"], index=df_freq.columns), ignore_index=True) 
        df_freq = df_freq.append(pd.Series([-1, "None", "IR"], index=df_freq.columns), ignore_index=True)
        df_freq = df_freq.append(pd.Series([-1, "None", "Vib.Temp."], index=df_freq.columns), ignore_index=True)
        df_freq = df_freq.append(pd.Series([-1, "None", "E_vib(kJ/mol)"], index=df_freq.columns), ignore_index=True)
        df_freq = df_freq.append(pd.Series([-1, "None", "S_vib(kJ/mol-K)"], index=df_freq.columns), ignore_index=True)
        for i in molDATA["atomID"].values:
            for xyz in ["X","Y","Z"]:
                df_freq = df_freq.append(pd.Series([i, xyz, "Mode"], index=df_freq.columns), ignore_index=True) 
        
        for f in col_data["freq"]:
            if "Thermochemistry" in f:
                break
            fs = f.split()
            if (not("--" in f))and(len(fs)<4):
                try:
                    for i in fs:
                        df_freq[str(int(i))] = 0.
                    fm_id = fs
                except:
                    pass
            elif "Frequencies --" in f:
                for fid, i  in zip(fm_id, fs[2:]):
                    Tem = otherDATAs["Temperature(K)"]
                    fv = float(i) ; vt = fv*1.438775 ; vtT = vt/Tem 
                    df_freq.loc[df_freq["Value"]=="Freq.", fid] = fv
                    df_freq.loc[df_freq["Value"]=="Vib.Temp.", fid] = vt
                    df_freq.loc[df_freq["Value"]=="E_vib(kJ/mol)", fid]                                             = 8.31446 * Tem / 1000 * (vtT * (0.5+np.exp(-1*vtT)/(1-np.exp(-1*vtT))))
                    df_freq.loc[df_freq["Value"]=="S_vib(kJ/mol-K)", fid]                                             = 8.31446 / 1000 * (vtT*np.exp(-1*vtT)/(1-np.exp(-1*vtT))-np.log(1-np.exp(-1*vtT)))
                    
            elif "Red. masses --" in f:
                for fid, i  in zip(fm_id, fs[3:]):
                    df_freq.loc[df_freq["Value"]=="Mass", fid] = float(i)
            elif "Frc consts  --" in f:
                for fid, i  in zip(fm_id, fs[3:]):
                    df_freq.loc[df_freq["Value"]=="Force", fid] = float(i)
            elif "IR Inten    --" in f:
                for fid, i  in zip(fm_id, fs[3:]):
                    df_freq.loc[df_freq["Value"]=="IR", fid] = float(i)
            else:
                try:
                    atm_id = int(fs[0])
                    for itr in range(int((len(fs)-2)/3)):
                        for i, x in enumerate(["X","Y","Z"]):
                            df_freq.loc[(df_freq["atomID"]==atm_id)&(df_freq["xyz"]==x), fm_id[itr]] = float(fs[2 + i + 3*itr])
                except:
                    continue
       
    if printLog:
        print("FREQ finished.")
        print("elapsed_time:{0}".format(time.time() - start) + "[sec]")    
        
    return 1, molDATA, pathDATA, df_gp, df_moc, df_freq, otherDATAs




def Hydrogen_Bond_Searcher(method, basis, solvent, num_water, CPU_num=10, shareCore=1,calc_num=30,Opt_only=False,Chk_keep=False):
    HMR_file = ""
    for n_file in os.listdir(".//"):
        if n_file.lower().endswith('.log') or n_file.lower().endswith('.out'):
            HMR_file = n_file
            break
                 
    if not HMR_file:
        print("There is no start file.")
        return 0
    
    H_name, H_ext = os.path.splitext(HMR_file)
    
    if not(("open" in H_name)or("close" in H_name)):
        print("This file may not be HMR.")
        return 0
    
    _, mol, _, _, _, _, od = Read_GaussianOutput(".//" + HMR_file, gf_col=False, moc_col=False, freq_col=False)                          
    mol, dis, con, hbon, odH = HMR_Identify(mol, H_name) 
    mol_HMR = mol[~(mol["pos"]=="water")]
    
    #To make basic file
    if Opt_only:
        input_root = "# opt " + method.lower() + "/" + basis.lower()
    else:
        input_root = "# opt freq " + method.lower() + "/" + basis.lower()
    if solvent.lower() == "gas":
        input_root += " \n"
    else:
        input_root += " scrf=(solvent=water,"+ solvent.lower() +") \n"
    input_root += "integral=grid=ultrafine " + "\n\nHydrogen_Bond_Search\n\n"
    input_root += str(od["netCharge"])+ " " + str(od["spin"]) + "\n"
    
    input_atm = ""
    for i in range(len(mol_HMR)):
        a = mol_HMR.iloc[i]
        input_atm += " "+str(a["atomTYPE"])+ "      " + str(a["x"])+ "   " + str(a["y"])+ "   " + str(a["z"])+ "\n"
    
    
    HBond_formation = {}
    Gs = {}  ##double dict
    DISs = {}  ##double dict
    retryJOB = False
    
    with open(".//log.txt", "a") as l:
        l.write("\n----[Logs]------------------------------\n\n")
    with open(".//active_log.txt", "a") as l:
        l.write("\n----[Active Logs]------------------------------\n\n")
    
    if not os.path.exists(".//input_files"):
        os.makedirs(".//input_files")
    if not os.path.exists(".//chk_files"):
        os.makedirs(".//chk_files")
    if not os.path.exists(".//output_files"):
        os.makedirs(".//output_files")
    else:   ##for continue
        retryJOB = True
        fileNOs = []
        file_num = 0
        for n_dir in os.listdir(".//output_files"):
            if not "." in n_dir: ## if n_dir is dir
                try:
                    k = int(n_dir)  ## if n_dir is 1,2,3,4,...
                    Gs[k] = {} ; DISs[k] = {}
                    for n_file in os.listdir(".//output_files//"+n_dir):
                        if ".csv" in n_file:
                            HBond_formation[k] = pd.read_csv(".//output_files//"+n_dir+"//"+n_file, index_col=0).values
                        if n_file.lower().endswith('.log') or n_file.lower().endswith('.out'):
                            name, ext = os.path.splitext(n_file)
                            fileNOs.append(int(name.split("_")[1]))
                            ok, mol_dat, _, _, _, _, od = Read_GaussianOutput(".//output_files//"+n_dir+"//"+n_file,                                                                                     gf_col=False, moc_col=False, freq_col=False) 
                            mol_dat, dis, con, hbon, odH = HMR_Identify(mol_dat, H_name)
                            
                            if (od["method"].lower().lstrip("r")!=method.lower())or(od["basisSet"].lower()!=basis.lower())or                                                (od["solvent"].lower()!=solvent.lower())or(odH["water"]!=num_water):
                                print("CANNOT retry: condition is different.")
                                return 0
                            
                            DISs[k][name] = dis
                            if Opt_only:
                                Gs[k][name] = od["E(HF)"]
                            else:
                                Gs[k][name] = od["G"]
                except:  ## if n_dir is dismisses
                    for n_file in os.listdir(".//output_files//"+n_dir):
                        name, ext = os.path.splitext(n_file)
                        fileNOs.append(int(name.split("_")[1]))
            else:   ## if n_dir is file
                name, ext = os.path.splitext(n_dir)
                fileNOs.append(int(name.split("_")[1]))
                file_num += 1
                    
        
    #To make water placed files
    Duration = {}
    fileNO = 1
    if retryJOB: 
        fileNO += np.array(fileNOs, dtype=int).max()
    
    def Job_Launch(fno, retry=False, HMRs=mol_HMR): 
        
        fileNAME = "Job_"+str(fno)
        
        with open(".//input_files//"+fileNAME+".gjf", "w") as g:
            g.write("%nprocshared="+str(shareCore)) ; g.write("\n")
            g.write("%mem="+str(shareCore)+"000MB") ; g.write("\n")
            g.write("%chk=./chk_files/"+fileNAME+".chk") ; g.write("\n")
            
            if retry:
                g.write(input_root)
                for i in range(len(HMRs)):
                    a = HMRs.iloc[i]
                    g.write(" "+str(a["atomTYPE"])+ "      " + str(a["x"])+ "   " + str(a["y"])+ "   " + str(a["z"])+ "\n")
            else:
                g.write(input_root + input_atm)

                waters = []
                hetero = HMRs.loc[(HMRs["atomTYPE"]=="O")|(HMRs["atomTYPE"]=="N")|(HMRs["atomTYPE"]=="S")]
                hb_cand = hetero.loc[(~(hetero["bond"]==4)), ["id","atomTYPE","x","y","z"]].values.tolist()
                hb_cand_h = []
                for h in HMRs.loc[HMRs["atomTYPE"]=="H",["id","atomTYPE","x","y","z"]].values.tolist():
                    for hc in hetero["id"].values.tolist():
                        if con[h[0],hc]:
                            hb_cand_h.append(h)

                hb_cand = hb_cand + hb_cand_h

                while not(len(waters) == 3*num_water):
                    go = False
                    while (not go):
                        z = np.random.rand()*2-1 ; psy = (np.random.rand()*2-1)*np.pi
                        vec = np.array([np.sqrt(1-z**2)*np.cos(psy), np.sqrt(1-z**2)*np.sin(psy), z])
                        hb_target = random.choice(hb_cand)
                        if hb_target[1]=="H":
                            wo = np.round(np.array(hb_target[2:]) + 1.5*vec, 10)
                        else:
                            wo = np.round(np.array(hb_target[2:]) + 2.7*vec, 10)
                        if np.linalg.norm(HMRs.loc[:, ["x","y","z"]].values - wo, axis=1).min() > 2. :
                            if (not waters)or(np.linalg.norm(np.array(waters) - wo, axis=1).min() > 2.) :
                                go = True                               

                    h1go = False
                    while (not h1go):
                        if hb_target[1]=="H":
                            z1 = np.random.rand()*2-1 ; psy1 = (np.random.rand()*2-1)*np.pi
                            vec1 = np.array([np.sqrt(1-z1**2)*np.cos(psy1), np.sqrt(1-z1**2)*np.sin(psy1), z1])
                            wh1 = np.round(wo + vec1, 10)
                            if np.linalg.norm(HMRs.loc[:, ["x","y","z"]].values - wh1, axis=1).min() > 1.2 :
                                if (not waters)or(np.linalg.norm(np.array(waters) - wh1, axis=1).min() > 1.2) :
                                    h1go = True             
                        else:
                            vec1 = (np.array(hb_target[2:]) - wo) / np.linalg.norm(np.array(hb_target[2:]) - wo)
                            wh1 = np.round(wo + vec1, 10)
                            h1go = True 


                    h2go = False
                    itr = 0
                    while (not h2go):
                        itr += 1
                        z2 = np.random.rand()*2-1 ; psy2 = (np.random.rand()*2-1)*np.pi
                        vec2 = np.array([np.sqrt(1-z2**2)*np.cos(psy2), np.sqrt(1-z2**2)*np.sin(psy2), z2])
                        wh2 = np.round(wo + vec2, 10)
                        if (np.inner(vec1,vec2)<0)and(np.inner(vec1,vec2)>-0.5):
                            if np.linalg.norm(HMRs.loc[:, ["x","y","z"]].values - wh2, axis=1).min() > 1.2 :
                                if (not waters)or(np.linalg.norm(np.array(waters) - wh2, axis=1).min() > 1.2) :
                                    h2go = True
                                    waters.append(wo) ; waters.append(wh1) ; waters.append(wh2) 
                                    hb_cand.append([-1, "O"] + wo.tolist())
                                    hb_cand.append([-1, "H"] + wh1.tolist())
                                    hb_cand.append([-1, "H"] + wh2.tolist())
                                    g.write(f" O      {wo[0]:f}   {wo[1]:f}   {wo[2]:f}\n")
                                    g.write(f" H      {wh1[0]:f}   {wh1[1]:f}   {wh1[2]:f}\n")
                                    g.write(f" H      {wh2[0]:f}   {wh2[1]:f}   {wh2[2]:f}\n")
                        if itr > 30:
                            h2go = True
                
            g.write("\n\n")
        
        command = f"\"g09 < ./input_files/{fileNAME}.gjf > ./output_files/{fileNAME}.log &\""
        result = subprocess.run(f"bsub -n {shareCore} {command}", shell=True)
        if result.returncode != 0:
            print("Calculation "+str(fno)+ " had an error.")
            return 0
            
        return 1
    
    if not os.path.exists(".//output_files//dismissed"):
        os.makedirs(".//output_files//dismissed")
    
        
    if not retryJOB:
        with open(".//log.txt", "a") as l:
            l.write("This is brand new calculation.\n")
            l.write(str(int(CPU_num//shareCore))+" jobs are submitted.\n\n")
        for i in range(int(CPU_num//shareCore)):
            Job_Launch(fileNO)
            fileNO += 1
    else:
        with open(".//log.txt", "a") as l:
            l.write("This is retry calculation.\n")
            l.write("Current on-going file num. is " + str(file_num) + "\n")
            l.write("Current MAX file NO is " + str(fileNO-1) + "\n\n")
            
    
    indAll = [] ; indH = [] ; indW = [] ; nodes = []
    for i, cn in enumerate(hbon.columns):
        c = cn.split("_")
        nodes.append(c[0]+"\n"+c[1])
        if "water" in cn:
            indW.append(i)
        else:
            indH.append(i)
    for perW in itertools.permutations(indW):
        indAll.append(indH + list(perW))
        
    def same_HB(hb_base, hb_test):
        for indx in indAll:
            if (hb_base==hb_test[:,indx][indx,:]).all():
                return True        
        return False
    
    Search_END = False
    span = 300
    successFile = 0
    first_check = True
    while (not Search_END):
        
        if first_check and retryJOB:
            first_check = False
        else:
            time.sleep(span)
            
        with open(".//active_log.txt", "a") as l:
            l.write("File checked at " + str(datetime.datetime.now()) + "\n")
        
        for n_file in os.listdir("."):
            if "core." in n_file:
                os.remove(".//"+n_file)
            if "terminate" in n_file.lower():
                with open(".//active_log.txt", "a") as l:
                    l.write("Terminate file is found.\n")
                Search_END = True
            
        
        for n_file in os.listdir(".//output_files"):
            if n_file.lower().endswith('.log') or n_file.lower().endswith('.out'):
                name, ext = os.path.splitext(n_file)            
                ok, mol_dat, _, _, _, _, od = Read_GaussianOutput(".//output_files//" + n_file,                                                                                 gf_col=False, moc_col=False, freq_col=False)                
                
                if name in Duration.keys():
                    Duration[name] += 1
                else:
                    Duration[name] = 1
                
                if ok==1:
                    if (not Chk_keep) and os.path.isfile(".//chk_files//"+name+".chk"):
                        os.remove(".//chk_files//"+name+".chk")
                    mol_dat, dis, con, hbon, odH = HMR_Identify(mol_dat, H_name)
                    successFile += 1
                    if Opt_only:
                        ENE = od["E(HF)"]
                    else:
                        ENE = od["G"]
                    hbf = hbon.values
                    classified = False
                    for fid in HBond_formation.keys():
                        if same_HB(HBond_formation[fid], hbf):
                            classified = True
                            dismissed = False
                            for d in DISs[fid].keys():
                                if (np.abs(DISs[fid][d]-dis).max() < 0.1)and(np.abs(ENE-Gs[fid][d])<0.0001):
                                    dismissed = True
                                    if os.path.isfile(".//output_files//"+n_file):
                                        shutil.move(".//output_files//"+n_file, ".//output_files//dismissed//")
                                    with open(".//log.txt", "a") as l:
                                        l.write(name + " is dismissed : the same structure as " + str(d) +"\n")
                                    break
                                    
                            if not dismissed:
                                Gs[fid][name] = ENE
                                DISs[fid][name] = dis
                                if os.path.isfile(".//output_files//"+n_file):
                                    shutil.move(".//output_files//"+n_file, ".//output_files//"+str(fid)+"//")
                                with open(".//log.txt", "a") as l:
                                    l.write(name + " is classified : Group " + str(fid) +"\n")
                                break
                    
                    if not classified:
                        if HBond_formation:
                            k = np.array(list(HBond_formation.keys()), dtype=int).max()+1
                        else:
                            k = 1
                        HBond_formation[k] = hbf
                        Gs[k] = {name:ENE}
                        DISs[k] = {name:dis}
                        os.makedirs(".//output_files//"+str(k))
                        if os.path.isfile(".//output_files//"+n_file):
                            shutil.move(".//output_files//"+n_file, ".//output_files//"+str(k))
                        hbon.to_csv(".//output_files//"+str(k) + "//HBond"+str(k)+".csv")
                        fig,ax = plt.subplots()
                        G = nx.Graph()
                        G.add_nodes_from(nodes)
                        edges = []
                        for hi, hv  in enumerate(hbf):
                            for wi, wv in enumerate(hv):
                                if(wv): edges.append((nodes[hi], nodes[wi]))
                        G.add_edges_from(edges)
                        pos = nx.circular_layout(G)
                        nx.draw_networkx(G, pos, node_color="w", edge_color="c", with_labels=True)
                        ax.tick_params(bottom=False,left=False,right=False,top=False)
                        ax.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
                        if not os.path.exists(".//HBonds"):
                            os.makedirs(".//HBonds")
                        fig.savefig(".//HBonds//HBond_"+str(k)+".png")
                        with open(".//log.txt", "a") as l:
                            l.write(name + " is newly classified : Group " + str(k) +"\n")
                        
                elif ok==-2:
                    if os.path.isfile(".//output_files//"+n_file):
                        os.remove(".//output_files//"+n_file)
                    if os.path.isfile(".//chk_files//"+name+".chk"):
                        os.remove(".//chk_files//"+name+".chk")
                    if os.path.isfile(".//input_files//"+name+".gjf"):
                        os.remove(".//input_files//"+name+".gjf")
                    with open(".//log.txt", "a") as l:
                        l.write(name + " is restarted : coodination error. \n")                        
                    Job_Launch(name.split("_")[1], retry=True, HMRs=mol_dat)                   
                
                elif ok==-1:
                    if (not Chk_keep) and os.path.isfile(".//chk_files//"+name+".chk"):
                        os.remove(".//chk_files//"+name+".chk")
                    if os.path.isfile(".//output_files//"+n_file):
                        shutil.move(".//output_files//"+n_file, ".//output_files//dismissed//")
                    with open(".//log.txt", "a") as l:
                        l.write(name + " is dismissed : optimization failed. \n")
                    
                        
                elif Duration[name]*span > 172800:
                    subprocess.run("kill "+str(od["JobID"]), shell=True)
                    if (not Chk_keep) and os.path.isfile(".//chk_files//"+name+".chk"):
                        os.remove(".//chk_files//"+name+".chk")
                    if os.path.isfile(".//output_files//"+n_file):
                        shutil.move(".//output_files//"+n_file, ".//output_files//dismissed//")
                    with open(".//log.txt", "a") as l:
                        l.write(name + " is dismissed : optimization has not been complete for 2 days. \n")
                    
        
        file_num = 0
        for n_file in os.listdir(".//output_files"):
            if n_file.lower().endswith('.log') or n_file.lower().endswith('.out'):
                file_num += 1
                
        if (file_num == 0)and(successFile >= calc_num):
            with open(".//log.txt", "a") as l:
                l.write("Current file num. is " + str(file_num)+ "\n")
                l.write("Calculation is successfully ended.\n")
            Search_END = True
        elif file_num < int(CPU_num//shareCore):
            with open(".//log.txt", "a") as l:
                l.write("Current file num. is " + str(file_num)+ "\n")
                l.write(str(int(CPU_num//shareCore)-file_num)+" jobs are submitted.\n")
            for i in range(int(CPU_num//shareCore)-file_num):
                Job_Launch(fileNO)
                fileNO += 1
            
    if Opt_only:
        summary = pd.DataFrame(index=[], columns=["HBond", "JobID", "E(HF)"])
    else:
        summary = pd.DataFrame(index=[], columns=["HBond", "JobID", "G"])
    
    for hk in Gs.keys():
        for gk in Gs[hk].keys():
            record = pd.Series([hk, gk, Gs[hk][gk]], index=summary.columns)
            summary = summary.append(record, ignore_index=True)
    
    summary.to_csv(".//Summary.csv")
    with open(".//log.txt", "a") as l:
        l.write("Calculation are successfully ended.\n\n")
    
    return summary




Hydrogen_Bond_Searcher("B3LYP", "6-31g(d)", "PCM", 3, CPU_num=10, shareCore=1, calc_num=50)

