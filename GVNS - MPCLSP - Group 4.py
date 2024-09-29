#%%

import pandas as pd
import math
from tkinter import N
from typing import Set
from mip import Model, xsum, minimize, BINARY, INTEGER
import mip
import numpy as np
from random import random, randint
import time
import matplotlib.pyplot as plt
from sympy import product
import copy
import csv


def Time_Calculator(Time):
    Minutes = round(Time/60,0)
    Seconds = round((Time - Minutes * 60),0)
    FinalTime = "Computational Time: " + str(Minutes) + " minutes and " + str(Seconds) + " seconds.\n"
    return FinalTime



# INSTANCE GENERATOR FOR EACH CAPACITY


def Instance1():

    NPlants = 4
    NPeriods = 6
    NProducts = 15

    NewFile = pd.ExcelFile("C:\\Users\gui_t\\Desktop\\Instancia_MPCLSP2.xlsx")

    SetuCosts = pd.read_excel(NewFile,'Setup Cost')
    SetuCosts.to_numpy()
    TransferCosts = pd.read_excel(NewFile,'Transfer Cost')
    TransferCosts.to_numpy()
    InCosts = pd.read_excel(NewFile,'Inv Cost')
    InCosts.to_numpy()
    ProductioCosts = pd.read_excel(NewFile,'Production Cost')
    ProductioCosts.to_numpy()
    ProductioTime = pd.read_excel(NewFile,'Prod Time')
    ProductioTime.to_numpy()
    SetuTime = pd.read_excel(NewFile,'Setup Time')
    SetuTime.to_numpy()
    Demand1 = pd.read_excel(NewFile,'Demand1')
    Demand1.to_numpy()
    Demand2 = pd.read_excel(NewFile,'Demand2')
    Demand2.to_numpy()
    Demand3 = pd.read_excel(NewFile,'Demand3')
    Demand3.to_numpy()
    Demand4 = pd.read_excel(NewFile,'Demand4')
    Demand4.to_numpy()
    Demand5 = pd.read_excel(NewFile,'Demand5')
    Demand5.to_numpy()
    Demand6 = pd.read_excel(NewFile,'Demand6')
    Demand6.to_numpy()
    Capacity = pd.read_excel(NewFile,'Capacity')
    Capacity.to_numpy()

    NewFile.close()

    Demand = np.zeros((NProducts,NPlants,NPeriods))
    InvCosts = np.zeros(NProducts)
    SetupCosts = np.zeros((NProducts,NPlants))
    ProductionCosts = np.zeros((NProducts,NPlants))
    ProductionTime = np.zeros((NProducts,NPlants))
    SetupTime = np.zeros((NProducts,NPlants))
    OC = np.ones((NPlants,NPeriods))

    for i in range(NProducts):
        InvCosts[i] = InCosts[0][i]
        for j in range(NPlants):
            SetupCosts[i][j] = SetuCosts[j][i]
            ProductionCosts[i][j] = ProductioCosts[j][i]
            ProductionTime[i][j] = ProductioTime[j][i]
            SetupTime[i][j] = SetuTime[j][i]
            for t in range(NPeriods):
                if t == 0:
                    Demand[i][j][t] = Demand1[j][i]
                if t == 1:
                    Demand[i][j][t] = Demand2[j][i]
                if t == 2:
                    Demand[i][j][t] = Demand3[j][i]
                if t == 3:
                    Demand[i][j][t] = Demand4[j][i]
                if t == 4:
                    Demand[i][j][t] = Demand5[j][i]
                if t == 5:
                    Demand[i][j][t] = Demand6[j][i]

    
    
    return SetupCosts,TransferCosts,InvCosts,ProductionCosts,ProductionTime,SetupTime,Demand,Capacity,OC

def InstanceGenerator60(NProducts,NPlants,NPeriods,alpha):

    al = alpha

    SetupCosts = np.zeros((NProducts,NPlants))

    TransferCosts = np.zeros((NPlants,NPlants))

    InvCosts = np.zeros((NProducts))

    ProductionCosts = np.zeros((NProducts,NPlants))

    ProductionTime = np.zeros((NProducts,NPlants))

    SetupTime = np.zeros((NProducts,NPlants))

    Demand = np.zeros((NProducts,NPlants,NPeriods))

    Capacity = np.zeros((NPlants,NPeriods))
    
    OvertimeCost = np.zeros((NPlants,NPeriods))

    for i in range(NProducts):

        InvCosts[i] = round(np.random.uniform(0.2,0.4),2)

        for j in range(NPlants):

            SetupCosts[i][j] = round(np.random.uniform(50,950),0)
            ProductionCosts[i][j] = round(np.random.uniform(1.25,2.5),2) # Values rounded to the closest "standard" unit - viable?

            SetupTime[i][j] = round(np.random.uniform(10,50),0)
            ProductionTime[i][j] = round(np.random.uniform(1,5),0)

            for t in range(NPeriods):
                
                Demand[i][j][t] = round(np.random.uniform(40,180),0)

            for k in range(NPlants):
                TransferCosts[j][k] = round(np.random.uniform(0.2,0.4),2)

    
    for t in range(NPeriods):
        for j in range(NPlants):
            Capacity[j][t] = xsum(Demand[i][j][t] * ProductionTime[i][j] + SetupTime[i][j] for i in range(NProducts)) / 0.6  # 60% Occupancy (Demand/Production)   Do for 80% & 90%
            OvertimeCost[j][t] =  round(np.random.uniform(1,2),2)


    return SetupCosts,TransferCosts,InvCosts,ProductionCosts,ProductionTime,SetupTime,Demand,Capacity,OvertimeCost

def InstanceGenerator80(NProducts,NPlants,NPeriods,alpha):

    al = alpha

    SetupCosts = np.zeros((NProducts,NPlants))

    TransferCosts = np.zeros((NPlants,NPlants))

    InvCosts = np.zeros((NProducts))

    ProductionCosts = np.zeros((NProducts,NPlants))

    ProductionTime = np.zeros((NProducts,NPlants))

    SetupTime = np.zeros((NProducts,NPlants))

    Demand = np.zeros((NProducts,NPlants,NPeriods))

    Capacity = np.zeros((NPlants,NPeriods))

    OvertimeCost = np.zeros((NPlants,NPeriods))

    for i in range(NProducts):

        InvCosts[i] = round(np.random.uniform(0.2,0.4),2)

        for j in range(NPlants):

            SetupCosts[i][j] = round(np.random.uniform(50,950),0)
            ProductionCosts[i][j] = round(np.random.uniform(1.25,2.5),2) # Values rounded to the closest "standard" unit - viable?

            SetupTime[i][j] = round(np.random.uniform(10,50),0)
            ProductionTime[i][j] = round(np.random.uniform(1,5),0)

            for t in range(NPeriods):
                
                Demand[i][j][t] = round(np.random.uniform(40,180),0)

            for k in range(NPlants):
                TransferCosts[j][k] = round(np.random.uniform(0.2,0.4),2)

    
    for t in range(NPeriods):
        for j in range(NPlants):
            Capacity[j][t] = xsum(Demand[i][j][t] * ProductionTime[i][j] + SetupTime[i][j] for i in range(NProducts)) / 0.8  # 80% Occupancy (Demand/Production)   Do for 80% & 90%
            OvertimeCost[j][t] =  round(np.random.uniform(1,2),2)

    return SetupCosts,TransferCosts,InvCosts,ProductionCosts,ProductionTime,SetupTime,Demand,Capacity,OvertimeCost

def InstanceGenerator90(NProducts,NPlants,NPeriods,alpha):

    al = alpha

    SetupCosts = np.zeros((NProducts,NPlants))

    TransferCosts = np.zeros((NPlants,NPlants))

    InvCosts = np.zeros((NProducts))

    ProductionCosts = np.zeros((NProducts,NPlants))

    ProductionTime = np.zeros((NProducts,NPlants))

    SetupTime = np.zeros((NProducts,NPlants))

    Demand = np.zeros((NProducts,NPlants,NPeriods))

    Capacity = np.zeros((NPlants,NPeriods))

    OvertimeCost = np.zeros((NPlants,NPeriods))

    for i in range(NProducts):

        InvCosts[i] = round(np.random.uniform(0.2,0.4),2)

        for j in range(NPlants):

            SetupCosts[i][j] = round(np.random.uniform(50,950),0)
            ProductionCosts[i][j] = round(np.random.uniform(1.25,2.5),2) # Values rounded to the closest "standard" unit - viable?

            SetupTime[i][j] = round(np.random.uniform(10,50),0)
            ProductionTime[i][j] = round(np.random.uniform(1,5),0)

            for t in range(NPeriods):
                
                Demand[i][j][t] = round(np.random.uniform(40,180),0)

            for k in range(NPlants):
                TransferCosts[j][k] = round(np.random.uniform(0.2,0.4),2)

    
    for t in range(NPeriods):
        for j in range(NPlants):
            Capacity[j][t] = xsum(Demand[i][j][t] * ProductionTime[i][j] + SetupTime[i][j] for i in range(NProducts)) / 0.9  # 90% Occupancy (Demand/Production)   Do for 80% & 90%
            OvertimeCost[j][t] =  round(np.random.uniform(1,2),2)

    return SetupCosts,TransferCosts,InvCosts,ProductionCosts,ProductionTime,SetupTime,Demand,Capacity,OvertimeCost

def InstanceGenerator100(NProducts,NPlants,NPeriods,alpha):

    al = alpha

    SetupCosts = np.zeros((NProducts,NPlants))

    TransferCosts = np.zeros((NPlants,NPlants))

    InvCosts = np.zeros((NProducts))

    ProductionCosts = np.zeros((NProducts,NPlants))

    ProductionTime = np.zeros((NProducts,NPlants))

    SetupTime = np.zeros((NProducts,NPlants))

    Demand = np.zeros((NProducts,NPlants,NPeriods))

    Capacity = np.zeros((NPlants,NPeriods))

    OvertimeCost = np.zeros((NPlants,NPeriods))

    for i in range(NProducts):

        InvCosts[i] = round(np.random.uniform(0.2,0.4),2)

        for j in range(NPlants):

            SetupCosts[i][j] = round(np.random.uniform(50,950),0)
            ProductionCosts[i][j] = round(np.random.uniform(1.25,2.5),2) # Values rounded to the closest "standard" unit - viable?

            SetupTime[i][j] = round(np.random.uniform(10,50),0)
            ProductionTime[i][j] = round(np.random.uniform(1,5),0)

            for t in range(NPeriods):
                
                Demand[i][j][t] = round(np.random.uniform(40,180),0)

            for k in range(NPlants):
                TransferCosts[j][k] = round(np.random.uniform(0.2,0.4),2)

    
    for t in range(NPeriods):
        for j in range(NPlants):
            Capacity[j][t] = xsum(Demand[i][j][t] * ProductionTime[i][j] + SetupTime[i][j] for i in range(NProducts)) / 1  # 100% Occupancy (Demand/Production)   Do for 80% & 90%
            OvertimeCost[j][t] =  round(np.random.uniform(1,2),2)

    return SetupCosts,TransferCosts,InvCosts,ProductionCosts,ProductionTime,SetupTime,Demand,Capacity,OvertimeCost



# INITIAL SOLUTION  - OBTAIN Y



def EOQINS(NPlants,NProducts,NPeriods, Demand, InvCosts, SetupCosts, ProductionTime,SetupTime,Capacity): # Já com instance generator

    SetupMatrix = np.zeros((NProducts,NPlants,NPeriods))
    Cap = np.zeros((NPlants,NPeriods))
    for j in range(NPlants):
        for t in range(NPeriods):
            Cap[j][t] = Capacity[j][t] / NProducts # Capacity per product (Average) - in time
        
    for i in range(NProducts):
        for j in range(NPlants):
            
            EOQ = math.sqrt(2 * SetupCosts[i][j] * xsum(Demand[i][j][t] for t in range(NPeriods))/InvCosts[i])
            
            if EOQ < round((Cap[j][t] - SetupTime[i][j])/ProductionTime[i][j],0):
                OnHand = EOQ
            else:
                OnHand = round((Cap[j][t] - SetupTime[i][j])/ProductionTime[i][j],0)
            
            SetupMatrix[i][j][0] = 1
            for t in range(1,NPeriods):
            
                if OnHand < Demand[i][j][t]:
                    OnHand += EOQ
                    SetupMatrix[i][j][t] = 1
                else:
                    OnHand -= Demand[i][j][t]

    return SetupMatrix

def SilverMealINS(NPlants,NProducts,NPeriods, Demand, InvCosts, SetupCosts):  # Já com Instance generator

    SetupMatrix = np.zeros((NProducts,NPlants,NPeriods))
   
    for i in range(NProducts):
        for j in range(NPlants):
            t = 0
            aux = 0
            SetupMatrix[i][j][0] = 1
            Cinit = SetupCosts[i][j]
            
            while t < NPeriods:
                HCosts = 0
                for l in range(aux,t+1):
                    HCosts += Demand[i][j][l] * InvCosts[i] * (l-aux)
                #if i == 2 and j == 0:
                #        print(HCosts)
                if t == aux + 1 and aux != 0:
                    Cinit = SetupCosts[i][j]
                
                C = ((SetupCosts[i][j] + HCosts)/(t-aux + 1))
                if  Cinit - C > 0: # If it decreases
                    Cinit = C
                else:
                    SetupMatrix[i][j][t] = 1
                    aux = t
                
                t += 1
    
    
    return SetupMatrix

def SilverMealINSC(NPlants,NProducts,NPeriods, Demand, InvCosts, SetupCosts, ProductionTime,SetupTime,Capacity):  # Já com Instance generator

    SetupMatrix = np.zeros((NProducts,NPlants,NPeriods))
    Cap = np.zeros((NPlants,NPeriods))
    for j in range(NPlants):
        for t in range(NPeriods):
            Cap[j][t] = Capacity[j][t] / NProducts # Capacity per product (Average) - in time # Capacity per product (Average) - in time

    for i in range(NProducts):
        for j in range(NPlants):
            t = 0
            aux = 0
            SetupMatrix[i][j][0] = 1
            Cinit = SetupCosts[i][j]
            Produc = 0
            while t < NPeriods:
                HCosts = 0
                for l in range(aux,t+1):
                    HCosts += Demand[i][j][l] * InvCosts[i] * (l-aux)
                    Produc += Demand[i][j][l]
                #if i == 2 and j == 0:
                #        print(HCosts)
                if t == aux + 1 and aux != 0:
                    Cinit = SetupCosts[i][j]
                
                C = ((SetupCosts[i][j] + HCosts)/(t-aux + 1))
                if C > Cap[j][t]:
                    SetupMatrix[i][j][t] = 1
                    aux = t
                    Produc = 0
                elif  Cinit - C > 0: # If it decreases
                    Cinit = C
                    
                else:
                    SetupMatrix[i][j][t] = 1
                    aux = t
                    Produc = 0
                

                t += 1
    
    
    return SetupMatrix

def Lot4Lot(NPlants,NProducts,NPeriods):

    # Generates Lot4Lot Initial solution (only 1's)

    return np.ones((NProducts,NPlants,NPeriods))

def RandomSetup(NPlants,NProducts,NPeriods):
    
    Y = np.zeros((NProducts,NPlants,NPeriods))

    for i in range(NProducts):
        for j in range(NPlants):
            for t in range(NPeriods):
                Y[i][j][t] = randint(0,1)
    
    return Y





#FEASIBILITY




def VerifyFeasibilityDem(ProductionMatrix,OnHand,Demand):

    NPlants = len(ProductionMatrix[0]) #Nbr of columns
    NPeriods = len(ProductionMatrix[0][0])    #Nbr of rows
    NProducts = len(ProductionMatrix)
    Feasibility = 1

#Tests Demand

    for i in range(NProducts):
        for j in range(NPlants):
            for t in range(NPeriods):
                if OnHand[i][j][t] < Demand[i][j][t]:
                    Feasibility = 0
    
    return Feasibility






# LSSOLVER - LP solver






def LSSolverINS(SetupMatrix,ProductionTime,SetupTime,SetupCosts,ProductionCosts,InvCosts,TransferCosts,Demand,Capacity):  #Applies to instance generated
    
    NPlants = len(SetupCosts[0]) #Nbr of columns
    NPeriods = len(Capacity[0])    #Nbr of rows
    NProducts = len(SetupCosts)

    model = Model(solver_name="GRB")

    OvertimeWeight = 0.25

    X = [[[model.add_var(var_type=INTEGER) for t in range(NPeriods)] for j in range(NPlants)] for i in range(NProducts)]

    I = [[[model.add_var(var_type=INTEGER) for t in range(NPeriods+1)] for j in range(NPlants)] for i in range(NProducts)]

    T = [[[[model.add_var(var_type=INTEGER) for t in range(NPeriods)] for j in range(NPlants)] for k in range(NPlants)] for i in range(NProducts)]

    O = [[[model.add_var(var_type=INTEGER) for t in range(NPeriods)] for j in range(NPlants)] for i in range(NProducts)]

    model.objective = minimize(xsum(ProductionCosts[i][j]*X[i][j][t] + OvertimeWeight * O[i][j][t] + InvCosts[i]*I[i][j][t+1] + xsum(T[i][j][k][t] * TransferCosts[k][j] for k in range(NPlants)) for t in range(NPeriods) for j in range(NPlants) for i in range(NProducts)))



    # Inventory Balance

    for i in range(NProducts):
        for j in range(NPlants):
            for t in range(NPeriods):
                model += I[i][j][t] + X[i][j][t] - xsum(T[i][j][k][t] for k in range(NPlants)) + xsum(T[i][k][j][t] for k in range(NPlants)) - I[i][j][t+1] == Demand[i][j][t]

    # Setup balance

    for i in range(NProducts):
        for j in range(NPlants):
            for t in range(NPeriods):
                model += X[i][j][t] <= (xsum(Demand[i][j][t] for j in range(NPlants) for t in range(NPeriods))) * SetupMatrix[i][j][t]

    # Capacity

    for j in range(NPlants):
        for t in range(NPeriods):
            model += xsum(ProductionTime[i][j] * X[i][j][t] + SetupMatrix[i][j][t] * SetupTime[i][j] for i in range(NProducts)) <= Capacity[j][t] + xsum(O[i][j][t] for i in range(NProducts))

    # 0 Inv. at period 0

    for i in range(NProducts):
        for j in range(NPlants):
            model += I[i][j][0] == 0

    # Non-negativy

    for i in range(NProducts):
        for j in range(NPlants):
            for t in range(NPeriods):
                model += X[i][j][t] >= 0
                model += I[i][j][t+1] >= 0
                model += O[i][j][t] >= 0
                
    # Non-negativity (Transfers)

    for i in range(NProducts):
        for j in range(NPlants):
            for k in range(NPlants):
                for t in range(NPeriods):
                    model += T[i][j][k][t] >= 0


    # Optimize

    model.optimize(max_seconds=120)

    ModelSolution = model.objective_value


   # Obtain objective values of production, inventory & transfers

    ProductionMatrix = np.zeros((NPlants,NProducts,NPeriods))
    NewY = np.zeros((NProducts,NPlants,NPeriods))
    InventoryMatrix = np.zeros((NProducts,NPlants,NPeriods))
    TransferMatrix = np.zeros((NProducts,NPlants,NPeriods))
    OnHand = np.zeros((NProducts,NPlants,NPeriods))
    InventoryMatrix = np.zeros((NProducts,NPlants,NPeriods))
    TransferMatrix = np.zeros((NProducts,NPlants,NPeriods))
    TProductionCosts = 0
    OverTime = 0

    a =0
    b = 0
    c= 0

    #print(type(O[a][b][c].x))
    # Extracting solution to previous arrays
    if X[a][b][c].x is not None:
        for j in range(NPlants):
            for i in range(NProducts):
                for t in range(NPeriods):
                    OverTime += O[i][j][t].x
                    ProductionMatrix[j][i][t] = X[i][j][t].x
                    InventoryMatrix[i][j][t] = I[i][j][t].x
                    
                    if X[i][j][t].x > 0:
                        NewY[i][j][t] = 1
                    

        for i in range(NProducts):
            for j in range(NPlants):    
                for t in range(NPeriods):
                    TransferMatrix[i][j][t] = xsum(T[i][j][k][t].x for k in range(NPlants))                
                    OnHand[i][j][t] = I[i][j][t] + X[i][j][t] - xsum(T[i][j][k][t] for k in range(NPlants)) + xsum(T[i][k][j][t] for k in range(NPlants)) - I[i][j][t+1]

        ExtraSetupCosts = xsum(NewY[i][j][t] * SetupCosts[i][j] for i in range(NProducts) for j in range(NPlants) for t in range(NPeriods))
        TInvCosts = xsum(InvCosts[i] * InventoryMatrix[i][j][t] for i in range(NProducts) for j in range(NPlants) for t in range(NPeriods))
        TTransferCosts = xsum(T[i][j][k][t].x * TransferCosts[j][k] for i in range(NProducts) for j in range(NPlants) for k in range(NPlants) for t in range(NPeriods))
        TProductionCosts = xsum(ProductionCosts[i][j] * X[i][j][t].x  for i in range(NProducts) for j in range(NPlants) for t in range(NPeriods))

        ModelSolution = ModelSolution + ExtraSetupCosts
        
        #print(TransferMatrix)
        #print(InventoryMatrix)
        #print(ProductionMatrix)

        #Printing Results

        #print("Results:\n")
        #print("Transfer Costs: " + str(round(float(TTransferCosts),2)))
        #print("Inventory Costs: " + str(round(float(TInvCosts),2)))
        #print("Setup Costs: " + str(round(float(ExtraSetupCosts),2)))
        #print("Model Solution: " + str(round(float(ModelSolution),2)))
        #print("Overtime: " + str(OverTime) + "\n")

        TTransferCosts = round(float(TTransferCosts),2)
        TInvCosts = round(float(TInvCosts),2)
        TProductionCosts = round(float(TProductionCosts),2)
        ExtraSetupCosts = round(float(ExtraSetupCosts),2)
        TotalC = round(float(ModelSolution),2)

        return NewY, TotalC, TTransferCosts, TInvCosts , ExtraSetupCosts, OverTime, ProductionMatrix , OnHand, TProductionCosts

        
    else:
        
        M=9999999
        return SetupMatrix,M,M,M,M,M,M,M,M

def LSSolverINSC(SetupMatrix,ProductionTime,SetupTime,SetupCosts,ProductionCosts,InvCosts,TransferCosts,Demand,Capacity,OC):  #Applies to instance generated
    
    NPlants = len(SetupCosts[0]) #Nbr of columns
    NPeriods = len(Capacity[0])    #Nbr of rows
    NProducts = len(SetupCosts)

    model = Model(solver_name="GRB")

    OvertimeWeight = 0.5

    X = [[[model.add_var(var_type=INTEGER) for t in range(NPeriods)] for j in range(NPlants)] for i in range(NProducts)]

    I = [[[model.add_var(var_type=INTEGER) for t in range(NPeriods+1)] for j in range(NPlants)] for i in range(NProducts)]

    T = [[[[model.add_var(var_type=INTEGER) for t in range(NPeriods)] for j in range(NPlants)] for k in range(NPlants)] for i in range(NProducts)]

    O = [[[model.add_var(var_type=INTEGER) for t in range(NPeriods)] for j in range(NPlants)] for i in range(NProducts)]

    model.objective = minimize(xsum(ProductionCosts[i][j] * X[i][j][t] + OvertimeWeight * O[i][j][t] + InvCosts[i]*I[i][j][t+1] + xsum(T[i][j][k][t] * TransferCosts[k][j] for k in range(NPlants)) for t in range(NPeriods) for j in range(NPlants) for i in range(NProducts)))

#OC[j][t]


    # Inventory Balance

    for i in range(NProducts):
        for j in range(NPlants):
            for t in range(NPeriods):
                model += I[i][j][t] + X[i][j][t] - xsum(T[i][j][k][t] for k in range(NPlants)) + xsum(T[i][k][j][t] for k in range(NPlants)) - I[i][j][t+1] == Demand[i][j][t]

    # Setup balance

    for i in range(NProducts):
        for j in range(NPlants):
            for t in range(NPeriods):
                model += X[i][j][t] <= (xsum(Demand[i][j][t] for j in range(NPlants) for t in range(NPeriods))) * SetupMatrix[i][j][t]

    # Capacity

    for j in range(NPlants):
        for t in range(NPeriods):
            model += xsum(ProductionTime[i][j] * X[i][j][t] + SetupMatrix[i][j][t] * SetupTime[i][j] for i in range(NProducts)) <= Capacity[j][t] + xsum(O[i][j][t] for i in range(NProducts))

    # 0 Inv. at period 0

    for i in range(NProducts):
        for j in range(NPlants):
            model += I[i][j][0] == 0

    # Non-negativy

    for i in range(NProducts):
        for j in range(NPlants):
            for t in range(NPeriods):
                model += X[i][j][t] >= 0
                model += I[i][j][t+1] >= 0
                model += O[i][j][t] >= 0
                
    # Non-negativity (Transfers)

    for i in range(NProducts):
        for j in range(NPlants):
            for k in range(NPlants):
                for t in range(NPeriods):
                    model += T[i][j][k][t] >= 0


    # Optimize

    model.optimize(max_seconds=120)

    ModelSolution = model.objective_value


   # Obtain objective values of production, inventory & transfers

    ProductionMatrix = np.zeros((NPlants,NProducts,NPeriods))
    NewY = np.zeros((NProducts,NPlants,NPeriods))
    InventoryMatrix = np.zeros((NProducts,NPlants,NPeriods))
    TransferMatrix = np.zeros((NProducts,NPlants,NPeriods))
    OnHand = np.zeros((NProducts,NPlants,NPeriods))
    InventoryMatrix = np.zeros((NProducts,NPlants,NPeriods))
    TransferMatrix = np.zeros((NProducts,NPlants,NPeriods))
    TProductionCosts = 0
    OverTime = 0

    a =0
    b = 0
    c= 0

    #print(type(O[a][b][c].x))
    # Extracting solution to previous arrays
    if X[a][b][c].x is not None:
        for j in range(NPlants):
            for i in range(NProducts):
                for t in range(NPeriods):
                    OverTime += O[i][j][t].x
                    ProductionMatrix[j][i][t] = X[i][j][t].x
                    InventoryMatrix[i][j][t] = I[i][j][t].x
                    
                    if X[i][j][t].x > 0:
                        NewY[i][j][t] = 1
                    

        for i in range(NProducts):
            for j in range(NPlants):    
                for t in range(NPeriods):
                    TransferMatrix[i][j][t] = xsum(T[i][j][k][t].x for k in range(NPlants))                
                    OnHand[i][j][t] = I[i][j][t] + X[i][j][t] - xsum(T[i][j][k][t] for k in range(NPlants)) + xsum(T[i][k][j][t] for k in range(NPlants)) - I[i][j][t+1]

        ExtraSetupCosts = xsum(NewY[i][j][t] * SetupCosts[i][j] for i in range(NProducts) for j in range(NPlants) for t in range(NPeriods))
        TInvCosts = xsum(InvCosts[i] * InventoryMatrix[i][j][t] for i in range(NProducts) for j in range(NPlants) for t in range(NPeriods))
        TTransferCosts = xsum(T[i][j][k][t].x * TransferCosts[j][k] for i in range(NProducts) for j in range(NPlants) for k in range(NPlants) for t in range(NPeriods))
        TProductionCosts = xsum(ProductionCosts[i][j] * X[i][j][t].x  for i in range(NProducts) for j in range(NPlants) for t in range(NPeriods))

        ModelSolution = ModelSolution + ExtraSetupCosts
        
        #print(TransferMatrix)
        #print(InventoryMatrix)
        #print(ProductionMatrix)

        #Printing Results

        #print("Results:\n")
        #print("Transfer Costs: " + str(round(float(TTransferCosts),2)))
        #print("Inventory Costs: " + str(round(float(TInvCosts),2)))
        #print("Setup Costs: " + str(round(float(ExtraSetupCosts),2)))
        #print("Model Solution: " + str(round(float(ModelSolution),2)))
        #print("Overtime: " + str(OverTime) + "\n")

        TTransferCosts = round(float(TTransferCosts),2)
        TInvCosts = round(float(TInvCosts),2)
        TProductionCosts = round(float(TProductionCosts),2)
        ExtraSetupCosts = round(float(ExtraSetupCosts),2)
        TotalC = round(float(ModelSolution),2)

        return NewY, TotalC, TTransferCosts, TInvCosts , ExtraSetupCosts, OverTime, ProductionMatrix , OnHand, TProductionCosts

        
    else:
        
        M=9999999
        return SetupMatrix,M,M,M,M,M,M,M,M

def LSSolverOptINS(ProductionTime,SetupTime,SetupCosts,ProductionCosts,InvCosts,TransferCosts,Demand,Capacity):

    NPlants = len(SetupCosts[0]) #Nbr of columns
    NPeriods = len(Capacity[0])    #Nbr of rows
    NProducts = len(SetupCosts)
    M = 99999999
    
    model = Model(solver_name="GRB")

    OvertimeWeight = 1000

    X = [[[model.add_var(var_type=INTEGER) for t in range(NPeriods)] for j in range(NPlants)] for i in range(NProducts)]

    I = [[[model.add_var(var_type=INTEGER) for t in range(NPeriods+1)] for j in range(NPlants)] for i in range(NProducts)]

    T = [[[[model.add_var(var_type=INTEGER) for t in range(NPeriods)] for j in range(NPlants)] for k in range(NPlants)] for i in range(NProducts)]

    O = [[[model.add_var(var_type=INTEGER) for t in range(NPeriods)] for j in range(NPlants)] for i in range(NProducts)]

    S = [[[model.add_var(var_type=BINARY) for t in range(NPeriods)] for j in range(NPlants)] for i in range(NProducts)]

    Z = [[[model.add_var(var_type=INTEGER) for t in range(NPeriods)] for j in range(NPlants)] for i in range(NProducts)] 

    model.objective = minimize(xsum(ProductionCosts[i][j]*X[i][j][t] + 1000 * O[i][j][t] + S[i][j][t] * SetupCosts[i][j] + InvCosts[i]*I[i][j][t+1] + xsum(T[i][j][k][t] * TransferCosts[k][j] for k in range(NPlants)) for t in range(NPeriods) for j in range(NPlants) for i in range(NProducts)))



    # Inventory Balance

    for i in range(NProducts):
        for j in range(NPlants):
            for t in range(NPeriods):
                model += I[i][j][t] + X[i][j][t] - xsum(T[i][j][k][t] for k in range(NPlants)) + xsum(T[i][k][j][t] for k in range(NPlants)) - I[i][j][t+1] == Demand[i][j][t]

    # Setup balance

    for i in range(NProducts):
        for j in range(NPlants):
            for t in range(NPeriods):
                model += Z[i][j][t] >= (xsum(X[i][j][t] for j in range(NPlants) for t in range(NPeriods))) - (1 - S[i][j][t]) * M
                model += Z[i][j][t] <= S[i][j][t] * M
                model += Z[i][j][t] >= 0
                model += Z[i][j][t] <= (xsum(Demand[i][j][t] for j in range(NPlants) for t in range(NPeriods)))
                model += Z[i][j][t] >= X[i][j][t]
                

    # Capacity

    for j in range(NPlants):
        for t in range(NPeriods):
            model += xsum(ProductionTime[i][j] * X[i][j][t] + S[i][j][t] * SetupTime[i][j] for i in range(NProducts)) <= Capacity[j][t] + xsum(O[i][j][t] for i in range(NProducts))

    # 0 Inv. at period 0

    for i in range(NProducts):
        for j in range(NPlants):
            model += I[i][j][0] == 0

    # Non-negativy

    for i in range(NProducts):
        for j in range(NPlants):
            for t in range(NPeriods):
                model += X[i][j][t] >= 0
                model += I[i][j][t+1] >= 0
                model += O[i][j][t] >= 0
                
    # Non-negativity (Transfers)

    for i in range(NProducts):
        for j in range(NPlants):
            for k in range(NPlants):
                for t in range(NPeriods):
                    model += T[i][j][k][t] >= 0


    # Optimize

    model.optimize(max_seconds=600)

    ModelSolution = model.objective_value


    # Obtain objective values of production, inventory & transfers

    ProductionMatrix = np.zeros((NPlants,NProducts,NPeriods))
    NewY = np.zeros((NProducts,NPlants,NPeriods))
    InventoryMatrix = np.zeros((NProducts,NPlants,NPeriods))
    TransferMatrix = np.zeros((NProducts,NPlants,NPeriods))
    OnHand = np.zeros((NProducts,NPlants,NPeriods))
    SMatrix = np.zeros((NProducts,NPlants,NPeriods))
    
    OverTime = 0


    #print(type(O[a][b][c].x))
    # Extracting solution to previous arrays
    print(model.status)
    if str(model.status) != "OptimizationStatus.INFEASIBLE":
        for j in range(NPlants):
            for i in range(NProducts):
                for t in range(NPeriods):
                    OverTime += O[i][j][t].x
                    ProductionMatrix[j][i][t] = X[i][j][t].x
                    InventoryMatrix[i][j][t] = I[i][j][t].x
                    SMatrix[i][j][t] = S[i][j][t].x

                    if X[i][j][t].x > 0:
                        NewY[i][j][t] = 1
                    

        for i in range(NProducts):
            for j in range(NPlants):    
                for t in range(NPeriods):
                    TransferMatrix[i][j][t] = xsum(T[i][j][k][t].x for k in range(NPlants))   
                    OnHand[i][j][t] = I[i][j][t].x + X[i][j][t].x - xsum(T[i][j][k][t].x for k in range(NPlants)) + xsum(T[i][k][j][t].x for k in range(NPlants)) - I[i][j][t+1].x
                    
        TInvCosts = xsum(InvCosts[i] * I[i][j][t].x for i in range(NProducts) for j in range(NPlants) for t in range(NPeriods))
        TTransferCosts = xsum(T[i][j][k][t].x * TransferCosts[j][k] for i in range(NProducts) for j in range(NPlants) for k in range(NPlants) for t in range(NPeriods))
        TSetupCosts = xsum(SetupCosts[i][j] * S[i][j][t].x for i in range(NProducts) for j in range(NPlants) for t in range(NPeriods))
        TProdCosts = xsum(ProductionCosts[i][j] * X[i][j][t].x for i in range(NProducts) for j in range(NPlants) for t in range(NPeriods))
        #print(TransferMatrix)
        #print(InventoryMatrix)
        #print(ProductionMatrix)

        #Printing Results

        print("Results:\n")
        print("Transfer Costs: " + str(round(float(TTransferCosts),2)))
        print("Inventory Costs: " + str(round(float(TInvCosts),2)))
        print("Setup Costs: " + str(round(float(TSetupCosts),2)))
        print("Production Costs: " + str(round(float(TProdCosts),2)))
        
        #print("Model Solution: " + str(round(float(ModelSolution),2)))
        print("Overtime: " + str(OverTime) + "\n")



        TTransferCosts = round(float(TTransferCosts),2)
        TInvCosts = round(float(TInvCosts),2)
        
        TotalC = round(float(ModelSolution),2)
       
        #print(SMatrix)
        return NewY, TotalC, TTransferCosts, TInvCosts , OverTime, ProductionMatrix, OnHand

def LSSolverOptINSC(ProductionTime,SetupTime,SetupCosts,ProductionCosts,InvCosts,TransferCosts,Demand,Capacity,OC):

    NPlants = len(SetupCosts[0]) #Nbr of columns
    NPeriods = len(Capacity[0])    #Nbr of rows
    NProducts = len(SetupCosts)
    M = 99999999
    
    model = Model(solver_name="GRB")

    OvertimeWeight = 1000

    X = [[[model.add_var(var_type=INTEGER) for t in range(NPeriods)] for j in range(NPlants)] for i in range(NProducts)]

    I = [[[model.add_var(var_type=INTEGER) for t in range(NPeriods+1)] for j in range(NPlants)] for i in range(NProducts)]

    T = [[[[model.add_var(var_type=INTEGER) for t in range(NPeriods)] for j in range(NPlants)] for k in range(NPlants)] for i in range(NProducts)]

    O = [[[model.add_var(var_type=INTEGER) for t in range(NPeriods)] for j in range(NPlants)] for i in range(NProducts)]

    S = [[[model.add_var(var_type=BINARY) for t in range(NPeriods)] for j in range(NPlants)] for i in range(NProducts)]

    Z = [[[model.add_var(var_type=INTEGER) for t in range(NPeriods)] for j in range(NPlants)] for i in range(NProducts)] 

    model.objective = minimize(xsum(ProductionCosts[i][j]*X[i][j][t] + OC[j][t] * O[i][j][t] + S[i][j][t] * SetupCosts[i][j] + InvCosts[i]*I[i][j][t+1] + xsum(T[i][j][k][t] * TransferCosts[k][j] for k in range(NPlants)) for t in range(NPeriods) for j in range(NPlants) for i in range(NProducts)))



    # Inventory Balance

    for i in range(NProducts):
        for j in range(NPlants):
            for t in range(NPeriods):
                model += I[i][j][t] + X[i][j][t] - xsum(T[i][j][k][t] for k in range(NPlants)) + xsum(T[i][k][j][t] for k in range(NPlants)) - I[i][j][t+1] == Demand[i][j][t]

    # Setup balance

    for i in range(NProducts):
        for j in range(NPlants):
            for t in range(NPeriods):
                model += Z[i][j][t] >= (xsum(X[i][j][t] for j in range(NPlants) for t in range(NPeriods))) - (1 - S[i][j][t]) * M
                model += Z[i][j][t] <= S[i][j][t] * M
                model += Z[i][j][t] >= 0
                model += Z[i][j][t] <= (xsum(Demand[i][j][t] for j in range(NPlants) for t in range(NPeriods)))
                model += Z[i][j][t] >= X[i][j][t]
                

    # Capacity

    for j in range(NPlants):
        for t in range(NPeriods):
            model += xsum(ProductionTime[i][j] * X[i][j][t] + S[i][j][t] * SetupTime[i][j] for i in range(NProducts)) <= Capacity[j][t] + xsum(O[i][j][t] for i in range(NProducts))

    # 0 Inv. at period 0

    for i in range(NProducts):
        for j in range(NPlants):
            model += I[i][j][0] == 0

    # Non-negativy

    for i in range(NProducts):
        for j in range(NPlants):
            for t in range(NPeriods):
                model += X[i][j][t] >= 0
                model += I[i][j][t+1] >= 0
                model += O[i][j][t] >= 0
                
    # Non-negativity (Transfers)

    for i in range(NProducts):
        for j in range(NPlants):
            for k in range(NPlants):
                for t in range(NPeriods):
                    model += T[i][j][k][t] >= 0


    # Optimize

    model.optimize(max_seconds=600)

    ModelSolution = model.objective_value


    # Obtain objective values of production, inventory & transfers

    ProductionMatrix = np.zeros((NPlants,NProducts,NPeriods))
    NewY = np.zeros((NProducts,NPlants,NPeriods))
    InventoryMatrix = np.zeros((NProducts,NPlants,NPeriods))
    TransferMatrix = np.zeros((NProducts,NPlants,NPeriods))
    OnHand = np.zeros((NProducts,NPlants,NPeriods))
    SMatrix = np.zeros((NProducts,NPlants,NPeriods))
    
    OverTime = 0


    #print(type(O[a][b][c].x))
    # Extracting solution to previous arrays
    print(model.status)
    if str(model.status) != "OptimizationStatus.INFEASIBLE":
        for j in range(NPlants):
            for i in range(NProducts):
                for t in range(NPeriods):
                    OverTime += O[i][j][t].x
                    ProductionMatrix[j][i][t] = X[i][j][t].x
                    InventoryMatrix[i][j][t] = I[i][j][t].x
                    SMatrix[i][j][t] = S[i][j][t].x

                    if X[i][j][t].x > 0:
                        NewY[i][j][t] = 1
                    

        for i in range(NProducts):
            for j in range(NPlants):    
                for t in range(NPeriods):
                    TransferMatrix[i][j][t] = xsum(T[i][j][k][t].x for k in range(NPlants))   
                    OnHand[i][j][t] = I[i][j][t].x + X[i][j][t].x - xsum(T[i][j][k][t].x for k in range(NPlants)) + xsum(T[i][k][j][t].x for k in range(NPlants)) - I[i][j][t+1].x
                    
        TInvCosts = xsum(InvCosts[i] * I[i][j][t].x for i in range(NProducts) for j in range(NPlants) for t in range(NPeriods))
        TTransferCosts = xsum(T[i][j][k][t].x * TransferCosts[j][k] for i in range(NProducts) for j in range(NPlants) for k in range(NPlants) for t in range(NPeriods))
        TSetupCosts = xsum(SetupCosts[i][j] * S[i][j][t].x for i in range(NProducts) for j in range(NPlants) for t in range(NPeriods))
        TProdCosts = xsum(ProductionCosts[i][j] * X[i][j][t].x for i in range(NProducts) for j in range(NPlants) for t in range(NPeriods))
        #print(TransferMatrix)
        #print(InventoryMatrix)
        #print(ProductionMatrix)

        #Printing Results

        print("Results:\n")
        print("Transfer Costs: " + str(round(float(TTransferCosts),2)))
        print("Inventory Costs: " + str(round(float(TInvCosts),2)))
        print("Setup Costs: " + str(round(float(TSetupCosts),2)))
        print("Production Costs: " + str(round(float(TProdCosts),2)))
        
        #print("Model Solution: " + str(round(float(ModelSolution),2)))
        print("Overtime: " + str(OverTime) + "\n")



        TTransferCosts = round(float(TTransferCosts),2)
        TInvCosts = round(float(TInvCosts),2)
        
        TotalC = round(float(ModelSolution),2)
       
        #print(SMatrix)
        return NewY, TotalC, TTransferCosts, TInvCosts , OverTime, ProductionMatrix, OnHand

#Neighbors



def AddSetup(Y):
    Find = 0
    Ya = copy.deepcopy(Y)
    while Find == 0:
        
        i = randint(0,len(Ya)-1)
        j = randint(0,len(Ya[0])-1)
        t = randint(0,len(Ya[0][0])-1)
        
        #print(len(Ya))
        #print(len(Ya[0]))
        #print(len(Ya[0][0]))
        #print(i)
        #print(j)
        #print(t)
        if Ya[i][j][t] == 0:
            Ya[i][j][t] = 1
            Find = 1
                 
    return Ya   
    
def RemoveSetup(Y):
    Find = 0
    Yr = copy.deepcopy(Y)
    while Find == 0:
        
        i = randint(0,len(Yr)-1)
        j = randint(0,len(Yr[0])-1)
        t = randint(0,len(Yr[0][0])-1)

        if Yr[i][j][t] == 1:
            Yr[i][j][t] = 0
            Find = 1
    
    return Yr

def TransferSetupPlant(Y):  #Transfer between plants
    Find = 0
    Yt = copy.deepcopy(Y)
    while Find == 0:

        i = randint(0,len(Yt)-1)
        j = randint(0,len(Yt[0])-1)
        t = randint(0,len(Yt[0][0])-1)

        j1 = randint(0,len(Yt[0])-1)
        
        if (Yt[i][j][t] != Yt[i][j1][t]) and (j != j1):
        
            #Switch values
            Intermediate = Yt[i][j][t]
            Yt[i][j][t] = Yt[i][j1][t]
            Yt[i][j1][t] = Intermediate

            Find = 1

    return Yt

def TransferSetupPeriod(Y):
    Find = 0
    Yt = copy.deepcopy(Y)
    while Find == 0:

        i = randint(0,len(Yt)-1)
        j = randint(0,len(Yt[0])-1)
        t = randint(0,len(Yt[0][0])-1)

        t1 = randint(0,len(Yt[0])-1)
        
        if (Yt[i][j][t] != Yt[i][j][t1]) and (t != t1):
        
            #Switch values
            Intermediate = Yt[i][j][t]
            Yt[i][j][t] = Yt[i][j][t1]
            Yt[i][j][t1] = Intermediate

            Find = 1


    return Yt


NeighborHoods = [RemoveSetup,AddSetup,TransferSetupPlant,TransferSetupPeriod]


# LOCAL SEARCH


def NewLocSearchINS(InitialSolution,TotalCosts, neighbortype, neighborhoodSize,Instance1Output):

    
    NIterations = 0
    OptimaCounter = 0
    FirstPeriod = 0
    #plt.plot(1,TotalCosts, c="blue")
    #plt.grid(True)
    #plt.title("Local Search for " + str(neighbortype), fontsize = 15, fontname="Times New Roman", fontweight = "bold",color="white")
    
    # Max Possible neighbors
    NIter = 0
    NProducts = len(InitialSolution)
    NPlants = len(InitialSolution[0])
    NPeriods = len(InitialSolution[0][0])
    NTotal = NProducts * NPlants * NPeriods
    
    for i in range(NProducts):
        for j in range(NPlants):
            for t in range(NPeriods):
                if InitialSolution[i][j][t] == 0:
                    NIter += 1   # Número de Adds que se pode fazer
                if InitialSolution[i][j][t] == 1 and t == 1:
                    FirstPeriod += 1

    if neighbortype == 1: # Add
        MaxIterations = NIter
        Kay = round(0.02 * MaxIterations)   # 10% of N
    elif neighbortype == 0: #Remove
        MaxIterations = (NTotal - NIter) - FirstPeriod
        Kay = round(0.02 * MaxIterations)   # 10% of N

    else: #Transfer
        MaxIterations = NTotal 
        Kay = round(0.02 * MaxIterations)   # 10% of N
    

    
    

    BetterSolution = 0

    
    CurrentCosts = TotalCosts
    NewCosts = TotalCosts
    NewTCosts = 0
    NewICosts = 0
    NewSCosts = 0
    NewPCosts = 0
    NewOCosts = 0

    Costs1 = []
    TrCosts1 = []
    ICosts1 = []
    SCosts1 = []
    PCosts1 = []
    OCosts1 = []      

    LocalBestSolution2 = InitialSolution


    Y1 = InitialSolution
    while NIterations < Kay:   #Set k of N 
        
        
        
        k = 0
        if neighbortype == 0:
            Removecounts = 0
            for i in range(NProducts):
                for j in range(NPlants):
                    for t in range(1,NPeriods):
                        if InitialSolution[i][j][t] == 1:
                            Removecounts += 1
            if Removecounts <= neighborhoodSize: 
                 # If there are less removes than the nbr of removes we want to make
                break

        for k in range(neighborhoodSize):
            if k == 0:
                Y1 = NeighborHoods[neighbortype](InitialSolution)
            else:
                Y1 = NeighborHoods[neighbortype](Y1)
        

        #Runs LP

        LPOutputs = LSSolverINS(Y1,Instance1Output[4],Instance1Output[5],Instance1Output[0],Instance1Output[3],Instance1Output[2],Instance1Output[1],Instance1Output[6],Instance1Output[7])

        print("Current costs: " + str(LPOutputs[1])) 
        if NIterations == 0: # If it's the first iteration:
            OptimaCounter = 1
            CurrentBest = copy.deepcopy(LPOutputs[0])
            CurrentCosts = LPOutputs[1]
            CurrentTCosts = LPOutputs[2]
            CurrentICosts = LPOutputs[3]
            CurrentSCosts = LPOutputs[4]
            CurrentPCosts = LPOutputs[8]
            CurrentOCosts = LPOutputs[5]
            #plt.scatter(OptimaCounter,LPOutputs[2],c="blue")
            #plt.scatter(OptimaCounter,LPOutputs[3],c="blue")
            #plt.scatter(OptimaCounter,LPOutputs[4],c="blue")
        else:
            if LPOutputs[1] < CurrentCosts:
                
                CurrentBest = copy.deepcopy(LPOutputs[0])
                CurrentCosts = LPOutputs[1]
                CurrentTCosts = LPOutputs[2]
                CurrentICosts = LPOutputs[3]
                CurrentSCosts = LPOutputs[4]
                CurrentPCosts = LPOutputs[8]
                CurrentOCosts = LPOutputs[5]
                #plt.scatter(OptimaCounter,LPOutputs[2],c="yellow")
                #plt.scatter(OptimaCounter,LPOutputs[3],c="green")
                #plt.scatter(OptimaCounter,LPOutputs[4],c="red")
        NIterations += 1
    
    if CurrentCosts < NewCosts:
        
        LocalBestSolution2 = copy.deepcopy(CurrentBest)
        NewCosts = CurrentCosts
        NewTCosts = CurrentTCosts
        NewICosts = CurrentICosts
        NewSCosts = CurrentSCosts
        NewPCosts = CurrentPCosts
        NewOCosts = CurrentOCosts

        #Costs1[OptimaCounter] = NewCosts
        #TrCosts1[OptimaCounter] = NewTCosts
        #ICosts1[OptimaCounter] = NewICosts
        #SCosts1[OptimaCounter] = NewSCosts
        #PCosts1[OptimaCounter] = NewPCosts
        #OCosts1[OptimaCounter] = NewOCosts

        OptimaCounter += 1
        #plt.scatter(OptimaCounter,NewCosts,c="blue")
        print("New Neighbor: " + str(NewCosts) + "\n")
    
        #print("Transfer costs:" + str(NewTCosts))
        #print("Inventory Costs: " + str(NewICosts))
        #print("Setup Costs: " + str(NewSCosts))
        #print("Production Costs: " + str(NewPCosts))
        #print("Overtime Costs: " + str(NewOCosts) + "\n")
        NIterations = 0
        BetterSolution = 1

    #if OptimaCounter > 1:
    #    plt.show()
    
    return LocalBestSolution2,NewCosts,BetterSolution, NewTCosts,NewICosts,NewSCosts,NewPCosts,NewOCosts



# VNS WITH VND MIXED


def VNSINS(InitialSolution,TotalCosts, neighborhoodSizeAdd,neighborhoodSizeRemove,neighborhoodTransfer,Instance1Output):
   
    Solutionx = InitialSolution # represents x
    TCosts1 = TotalCosts # represents f(x)
    FinalSolution = InitialSolution
    neighborhoodSize = 1
    IncumbentSolution = InitialSolution
    Bestx1 = InitialSolution

    Costs1 = []
    TrCosts1 = []
    ICosts1 = []
    SCosts1 = []
    PCosts1 = []
    OCosts1 = []

    TotalTCostsx1 = 0
    TotalICostsx1 = 0
    TotalSCostsx1 = 0
    TotalPCostsx1 = 0
    TotalOCostsx1 = 0

    TTCosts1 = TotalTCostsx1
    TICosts1 = TotalICostsx1
    TSCosts1 = TotalSCostsx1
    TPCosts1 = TotalPCostsx1
    TOCosts1 = TotalPCostsx1
    
    l = 0
    counter = 0
    OptCounter = 0
    NonOptCounter = 0
    StartTime = time.time()
    EndTime = time.time() + 60 * 40

    while time.time() < EndTime and NonOptCounter < 20:   # 30 minutes or 20 shakes with no better value
        k = 0
        l = 0
    
        while k < len(NeighborHoods):
            
            counter += 1
            print("Shake number: " + str(counter))
            
            test = 0
            if k == 1:
                neighborhoodSize = neighborhoodSizeAdd
            elif k == 0:
                neighborhoodSize = neighborhoodSizeRemove
            else:
                neighborhoodSize = neighborhoodTransfer
            if k == 0:
                Removecounts = 0
                for i in range(len(InitialSolution)):
                    for j in range(len(InitialSolution[0])):
                        for t in range(1,len(InitialSolution[0][0])):
                            if Solutionx[i][j][t] == 1:
                                Removecounts += 1
                if Removecounts <= neighborhoodSize:   # If there are less removes than the nbr of removes we want to make
                    break
            
            while test <= 10:
                for w in range(neighborhoodSize):
                    if w == 0:
                        IncumbentSolution = NeighborHoods[k](Solutionx) #x'
                    else:
                        IncumbentSolution = NeighborHoods[k](IncumbentSolution) #x'

                Outp = LSSolverINS(IncumbentSolution,Instance1Output[4],Instance1Output[5],Instance1Output[0],Instance1Output[3],Instance1Output[2],Instance1Output[1],Instance1Output[6],Instance1Output[7])
                if Outp[1] != 99999999:
                    TotalCostx1 = Outp[1] #TCosts1
                
                    break
                else:
                    test += 1
            if test >= 9:   #If no feasible solution was found, next neighbor
                print("No feasible solutions within this neighborhood. Jump to next one")
                NonOptCounter += 1
                k += 1
            else:
                l = 0
                it = counter
                filename = "%s.csv" % it
                f = open(filename,"w")
                writer = csv.writer(f,delimiter=",")
                writer.writerow(["Time","Total","Transfer","Inv","Setup","Prod","OT","Neighbor"])
                while l < len(NeighborHoods):
                    LocSearchOutput = NewLocSearchINS(IncumbentSolution,TotalCostx1,l,1,Instance1Output) #LOCAL SEARCH WITH NEIGHBOR SIZE 1
                    
                    if LocSearchOutput[2] == 0:  #MOVE
                        print("No better was found in neighbor " + str(l))
                        l += 1
                    elif LocSearchOutput[2] == 1: #Better neighbor was found
                        
                        #print("Before local: " + str(TotalCostx1))
                        #print("After local: " + str(LocSearchOutput[1]))
                        IncumbentSolution = copy.deepcopy(LocSearchOutput[0])
                        Bestx1 = copy.deepcopy(LocSearchOutput[0])
                        TotalCostx1 = LocSearchOutput[1]
                        TotalTCostsx1 = LocSearchOutput[3]
                        TotalICostsx1 = LocSearchOutput[4]
                        TotalSCostsx1 = LocSearchOutput[5]
                        TotalPCostsx1 = LocSearchOutput[6]
                        TotalOCostsx1 = LocSearchOutput[7]
                        ins = [time.time()-StartTime,LocSearchOutput[1],LocSearchOutput[3],LocSearchOutput[4],LocSearchOutput[5],LocSearchOutput[6],LocSearchOutput[7],l]
                        writer.writerow(ins)
                        l = 0
                    if time.time() > EndTime:
                        break
                f.close()
                #print("The best found is: " + str(TotalCostx1) + " for neighbor " + str(k)) 
                if TotalCostx1 < TCosts1:
                    Solutionx = copy.deepcopy(Bestx1)
                    FinalSolution = copy.deepcopy(Bestx1)
                    TCosts1 = TotalCostx1
                    TTCosts1 = TotalTCostsx1
                    TICosts1 = TotalICostsx1
                    TSCosts1 = TotalSCostsx1
                    TPCosts1 = TotalPCostsx1
                    TOCosts1 = TotalOCostsx1

                    

                    print("New Local Optima:" + str(TCosts1))
                    print("New Transfer Costs:" + str(TTCosts1))
                    print("New Inventory Costs:" + str(TICosts1))
                    print("New Setup Costs:" + str(TSCosts1) + "\n")
                    print("New Production Costs:" + str(TPCosts1))
                    print("New Overtime Costs:" + str(TOCosts1))
                    OptCounter += 1
                    NonOptCounter = 0
                    k = 0
                else:
                    NonOptCounter += 1
                    k += 1
                
            if time.time() > EndTime:
                break
        
            
    
    print("Final Value: " + str(TCosts1))
    print("Final Transfer Costs:" + str(TTCosts1))
    print("Final Inventory Costs:" + str(TICosts1))
    print("Final Setup Costs:" + str(TSCosts1) + "\n")
    print(Time_Calculator(time.time() - StartTime))
    print(FinalSolution)


def VNSINSN(InitialSolution,TotalCosts, neighborhoodSizeAdd,neighborhoodSizeRemove,neighborhoodTransfer,Instance1Output):
   
    Solutionx = InitialSolution # represents x
    TCosts1 = TotalCosts # represents f(x)
    FinalSolution = InitialSolution
    neighborhoodSize = 1
    IncumbentSolution = InitialSolution
    Bestx1 = InitialSolution

    Costs1 = []
    TrCosts1 = []
    ICosts1 = []
    SCosts1 = []
    PCosts1 = []
    OCosts1 = []

    TotalTCostsx1 = 0
    TotalICostsx1 = 0
    TotalSCostsx1 = 0
    TotalPCostsx1 = 0
    TotalOCostsx1 = 0

    TTCosts1 = TotalTCostsx1
    TICosts1 = TotalICostsx1
    TSCosts1 = TotalSCostsx1
    TPCosts1 = TotalPCostsx1
    TOCosts1 = TotalPCostsx1
    
    TotalCostx1 = TotalCosts

    l = 0
    counter = 0
    OptCounter = 0
    NonOptCounter = 0
    StartTime = time.time()
    EndTime = time.time() + 60 * 40

    while time.time() < EndTime and NonOptCounter < 20:   # 30 minutes or 20 shakes with no better value
        k = 0
        l = 0
    
        while k < len(NeighborHoods):
            
            test = 1

            if test >= 9:   #If no feasible solution was found, next neighbor
                print("No feasible solutions within this neighborhood. Jump to next one")
                NonOptCounter += 1
                k += 1
            else:
                l = 0
                it = counter
                filename = "%s.csv" % it
                f = open(filename,"w")
                writer = csv.writer(f,delimiter=",")
                writer.writerow(["Time","Total","Transfer","Inv","Setup","Prod","OT","Neighbor"])
                while l < len(NeighborHoods):
                    LocSearchOutput = NewLocSearchINS(IncumbentSolution,TotalCostx1,l,1,Instance1Output) #LOCAL SEARCH WITH NEIGHBOR SIZE 1
                    
                    if LocSearchOutput[2] == 0:  #MOVE
                        print("No better was found in neighbor " + str(l))
                        l += 1
                    elif LocSearchOutput[2] == 1: #Better neighbor was found
                        
                        #print("Before local: " + str(TotalCostx1))
                        #print("After local: " + str(LocSearchOutput[1]))
                        IncumbentSolution = copy.deepcopy(LocSearchOutput[0])
                        Bestx1 = copy.deepcopy(LocSearchOutput[0])
                        TotalCostx1 = LocSearchOutput[1]
                        TotalTCostsx1 = LocSearchOutput[3]
                        TotalICostsx1 = LocSearchOutput[4]
                        TotalSCostsx1 = LocSearchOutput[5]
                        TotalPCostsx1 = LocSearchOutput[6]
                        TotalOCostsx1 = LocSearchOutput[7]
                        ins = [time.time()-StartTime,LocSearchOutput[1],LocSearchOutput[3],LocSearchOutput[4],LocSearchOutput[5],LocSearchOutput[6],LocSearchOutput[7],l]
                        writer.writerow(ins)
                        l = 0
                    if time.time() > EndTime:
                        break
                f.close()
                #print("The best found is: " + str(TotalCostx1) + " for neighbor " + str(k)) 
                if TotalCostx1 < TCosts1:
                    Solutionx = copy.deepcopy(Bestx1)
                    FinalSolution = copy.deepcopy(Bestx1)
                    TCosts1 = TotalCostx1
                    TTCosts1 = TotalTCostsx1
                    TICosts1 = TotalICostsx1
                    TSCosts1 = TotalSCostsx1
                    TPCosts1 = TotalPCostsx1
                    TOCosts1 = TotalOCostsx1

                    

                    print("New Local Optima:" + str(TCosts1))
                    print("New Transfer Costs:" + str(TTCosts1))
                    print("New Inventory Costs:" + str(TICosts1))
                    print("New Setup Costs:" + str(TSCosts1) + "\n")
                    print("New Production Costs:" + str(TPCosts1))
                    print("New Overtime Costs:" + str(TOCosts1))
                    OptCounter += 1
                    NonOptCounter = 0
                    k = 0
                else:
                    NonOptCounter += 1
                    k += 1
                
            if time.time() > EndTime:
                break
        
            
    
    print("Final Value: " + str(TCosts1))
    print("Final Transfer Costs:" + str(TTCosts1))
    print("Final Inventory Costs:" + str(TICosts1))
    print("Final Setup Costs:" + str(TSCosts1) + "\n")
    print(Time_Calculator(time.time() - StartTime))
    print(FinalSolution)


#RUN - THIS IS WHERE YOU CHANGE THINGS

#Get Instance - tira de comentário a que se vai usar e mete as outras em comentário - NÃO ESQUECER DE METER NUMERO DE PLANTAS, PRODUTOS E PERIODOS NAS GERADAS - Ex: InstanceGenerator60(15,4,6,1) (o ultimo é alpha e nao é usado para nada)

#Instance1Output = Instance1()
#Instance1Output = InstanceGenerator60(15,4,6,1)
Instance1Output = InstanceGenerator80(30,4,12,1)
#Instance1Output = InstanceGenerator90(30,4,12,1)
#Instance1Output = InstanceGenerator100(15,4,6,1)



#Get Y - Corre o LP - tira de comentário a que se vai usar e mete as outras em comentário



#SMOutput = LSSolver(SilverMeal(4,15,6,Instance1Output[6],Instance1Output[2],Instance1Output[0]))
#SMOutput = LSSolver(EOQ(4,15,6,Instance1Output[6],Instance1Output[2],Instance1Output[0]))
#SMOutput = LSSolver(Lot4Lot(15,4,6)



SMOutput = LSSolverINS(SilverMealINS(4,30,12,Instance1Output[6],Instance1Output[2],Instance1Output[0]),Instance1Output[4],Instance1Output[5],Instance1Output[0],Instance1Output[3],Instance1Output[2],Instance1Output[1],Instance1Output[6],Instance1Output[7])
#SMOutput = LSSolverINS(EOQINS(4,15,6,Instance1Output[6],Instance1Output[2],Instance1Output[0],Instance1Output[4],Instance1Output[5],Instance1Output[7]),Instance1Output[4],Instance1Output[5],Instance1Output[0],Instance1Output[3],Instance1Output[2],Instance1Output[1],Instance1Output[6],Instance1Output[7])
#SMOutput = LSSolverINS(Lot4Lot(4,15,6),Instance1Output[4],Instance1Output[5],Instance1Output[0],Instance1Output[3],Instance1Output[2],Instance1Output[1],Instance1Output[6],Instance1Output[7])
#SMOutput = LSSolverINS(RandomSetup(4,15,6),Instance1Output[4],Instance1Output[5],Instance1Output[0],Instance1Output[3],Instance1Output[2],Instance1Output[1],Instance1Output[6],Instance1Output[7])

#SMOutput = LSSolverINSC(SilverMealINS(4,15,6,Instance1Output[6],Instance1Output[2],Instance1Output[0]),Instance1Output[4],Instance1Output[5],Instance1Output[0],Instance1Output[3],Instance1Output[2],Instance1Output[1],Instance1Output[6],Instance1Output[7],Instance1Output[8])
#SMOutput = LSSolverINSC(EOQINS(4,15,6,Instance1Output[6],Instance1Output[2],Instance1Output[0],Instance1Output[4],Instance1Output[5],Instance1Output[7]),Instance1Output[4],Instance1Output[5],Instance1Output[0],Instance1Output[3],Instance1Output[2],Instance1Output[1],Instance1Output[6],Instance1Output[7],Instance1Output[8])
#SMOutput = LSSolverINSC(Lot4Lot(4,15,6),Instance1Output[4],Instance1Output[5],Instance1Output[0],Instance1Output[3],Instance1Output[2],Instance1Output[1],Instance1Output[6],Instance1Output[7],Instance1Output[8])
#SMOutput = LSSolverINSC(RandomSetup(4,15,6),Instance1Output[4],Instance1Output[5],Instance1Output[0],Instance1Output[3],Instance1Output[2],Instance1Output[1],Instance1Output[6],Instance1Output[7],Instance1Output[8])

#print("Initial solution costs: " + str(SMOutput[1]))
#print(SilverMealINSC(4,15,6,Instance1Output[6],Instance1Output[2],Instance1Output[0],Instance1Output[4],Instance1Output[5],Instance1Output[7]))
#print(SilverMealINS(4,15,6,Instance1Output[6],Instance1Output[2],Instance1Output[0]))

#Time

#StartTime = time.time()

#Run Optimal

#print("Total Optimal costs: " + str(LSSolverOpt()[1]))
#print("Total Optimal Costs: " + str(LSSolverOptINS(Instance1Output[4],Instance1Output[5],Instance1Output[0],Instance1Output[3],Instance1Output[2],Instance1Output[1],Instance1Output[6],Instance1Output[7])[1]) + "\n")
#print("Total Optimal Costs: " + str(LSSolverOptINSC(Instance1Output[4],Instance1Output[5],Instance1Output[0],Instance1Output[3],Instance1Output[2],Instance1Output[1],Instance1Output[6],Instance1Output[7],Instance1Output[8])[1]) + "\n")
#print("Total Optimal Costs: " + str(LSSolverOpt()[1]) + "\n")

#TIME

#print(Time_Calculator(time.time() - StartTime))

#Run VNS with VND - VNS(Matriz obtida pelo LP, custos totais para comparação, Tamanho do neighbor remove, tamanho do neighbor add, tamanho dos neighbors transfer)


VNSINSN(SMOutput[0],SMOutput[1],1,1,1,Instance1Output)









#%%
