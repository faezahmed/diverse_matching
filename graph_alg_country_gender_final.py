import networkx as nx
import bellmanford as bf
import numpy as np
import miqp_country_gender as mcg

import scipy.stats as sp

import random
import matplotlib.pyplot as plt

def transfer_multiple_worker(cycle, G, matr_team_country_gender, wmatr_team_country, lambd1, lambd2):
    new_matr=np.copy(matr_team_country_gender)  
    for i in range(len(cycle)-1):       
        snode = cycle[i]
        dnode = cycle[i+1]       
        t_snode = int(snode[1 : snode.find('C')])
        t_dnode = int(dnode[1 : dnode.find('C')])        
        c_snode = int(snode[snode.find('C')+1 : snode.find('G')])
        c_dnode = int(dnode[dnode.find('C')+1 : dnode.find('G')])
        g_snode = int(snode[snode.find('G')+1 : len(snode)-3])
        g_dnode = int(dnode[dnode.find('G')+1 : len(dnode)-3])

        if((t_snode != t_dnode) and (c_snode == c_dnode) and g_snode == g_dnode):
            if((snode[-3:] == 'out')):
                new_matr[t_snode][c_snode][g_snode] -= 1
                new_matr[t_dnode][c_dnode][g_dnode] += 1
            #the following case should not happen
            elif((dnode[-3:] == 'out')):
                new_matr[t_snode][c_snode][g_snode] += 1
                new_matr[t_dnode][c_dnode][g_dnode] -= 1               
    matt_diff = new_matr - matr_team_country_gender
    while True:
        temp_matr = new_matr + matt_diff
        if (temp_matr >= 0).all():
            new_fitness = objective_function(temp_matr, temp_matr.shape[0], temp_matr.shape[1], temp_matr.shape[2], wmatr_team_country, lambd1, lambd2)
            old_fitness = objective_function(new_matr, new_matr.shape[0], new_matr.shape[1], new_matr.shape[2], wmatr_team_country, lambd1, lambd2)
            if new_fitness >= old_fitness:
                break
            else:
                new_matr = temp_matr
        else:
            break
    return new_matr
def worker_country_gender(matr_team_country_gender, wmatr_team_country, num_teams,num_countries, num_genders,lambd1, lambd2):
    st = 'T0C0G0inp'
    ed = 'T0C0G0out'  
    while(True):
        G = find_graph_weighted_gender(matr_team_country_gender, wmatr_team_country, num_teams,num_countries, num_genders,lambd1, lambd2)
        length, cycle, negative_cycle = bf.bellman_ford(G, st, ed, weight = 'weight')
        if negative_cycle is False:
            break      
        matr_team_country_gender = transfer_multiple_worker(cycle, G, matr_team_country_gender, wmatr_team_country, lambd1, lambd2)            
        if((length > -1e-10) & (negative_cycle==True)):
            print ("Small value", length)
            break
    return matr_team_country_gender
def find_graph_weighted_gender(matr_team_country_gender, wmatr_team_country, num_teams,num_countries, num_genders,lambd1, lambd2):
    G = nx.DiGraph()
    labels=find_node_labels_gated_gender(num_teams,num_countries,num_genders)
    G.add_nodes_from(labels)    
    for i in range(num_teams):
        for j in range(num_countries):           
            for k in range(num_genders):            
                src ='T'+str(i)+'C'+str(j)+'G'+str(k)+'inp'
                sink ='T'+str(i)+'C'+str(j)+'G'+str(k)+'out'
                if(i==0):
                    G.add_edge(src, sink, weight = 0)
                else:
                    G.add_edge(src, sink, weight = -2*lambd1-2*lambd2)
                #Loop over nodes within same team, same country but other genders
                other_genders=[indx for indx in range(num_genders)]
                other_genders.remove(k)
                for l in other_genders:
                    sink_other_gender ='T'+str(i)+'C'+str(j)+'G'+str(l)+'out'
                    if (i==0):
                        G.add_edge(src, sink_other_gender, weight = 0)
                    else:
                        G.add_edge(src, sink_other_gender, weight = -2*lambd1)
                #Loop over nodes within same team, same gender, but different country                
                other_countries = [indx for indx in range(num_countries)] 
                other_countries.remove(j)
                for l in other_countries:
                    sink_other_countrie ='T'+str(i)+'C'+str(l)+'G'+str(k)+'out'
                    if (i==0):
                        G.add_edge(src, sink_other_countrie, weight = 0)
                    else:
                        G.add_edge(src, sink_other_countrie, weight = -2*lambd2)
                #Loop over nodes within same team, different gender, and different country
                for p in range(num_countries):
                    for q in range(num_genders):
                        if p == j or q == k:
                            continue
                        else:
                            G.add_edge(src, 'T'+str(i)+'C'+str(p)+'G'+str(q)+'out', weight = 0)
                #Loop over nodes within same country and gender but other teams
                other_teams = [indx for indx in range(num_teams)]
                other_teams.remove(i) 
                for i2 in other_teams: 
                    sink_other_team = 'T'+str(i2)+'C'+str(j)+'G'+str(k)+'inp'
                    weight = 0
                    #if sink_other_team == 'T1C2G1inp' and sink == 'T2C2G1out':                  
                    if(matr_team_country_gender[i][j][k]>0):
                        num_i_j = sum(matr_team_country_gender[i][j][x] for x in range(num_genders))
                        num_i_k = sum(matr_team_country_gender[i][x][k] for x in range(num_countries))
                        num_i2_j = sum(matr_team_country_gender[i2][j][x] for x in range(num_genders))
                        num_i2_k = sum(matr_team_country_gender[i2][x][k] for x in range(num_countries))
                        if (i2 != 0 ):
                            weight += (wmatr_team_country[i2][j])+lambd1*(1+2*num_i2_j) + lambd2*(1+2*num_i2_k)
                        if (i != 0):
                            weight += (-wmatr_team_country[i][j])+lambd1*(1-2*num_i_j) + lambd2*(1-2*num_i_k)
                    else:
                        weight=1000000
                    G.add_edge(sink, sink_other_team, weight=weight)
    return G
def find_node_labels_gated_gender(num_teams,num_countries,num_genders):
    prestring='T'
    labels=[0 for i in range(2*num_teams*num_countries*num_genders)]
    for i in range(num_teams):
        for j in range(num_countries):
            for k in range(num_genders):
                nodelabel=prestring+str(i)+'C'+str(j)+'G'+str(k)+'inp'
                labels[2*i*num_countries*num_genders+2*j*num_genders+2*k]=nodelabel
                nodelabel=prestring+str(i)+'C'+str(j)+'G'+str(k)+'out'
                labels[2*i*num_countries*num_genders+2*j*num_genders+2*k+1]=nodelabel
            
    return labels
def objective_function(matr_team_country_gender, num_teams_plus_one, num_countries, num_genders, weights, lambd1, lambd2):
    team_country = [[0 for c in range(num_countries)] for t in range(num_teams_plus_one)]
    for t in range(num_teams_plus_one):
        for c in range(num_countries):
            team_country[t][c] = sum(matr_team_country_gender[t][c][g] for g in range(num_genders))
    team_gender = [[0 for g in range(num_genders)] for t in range(num_teams_plus_one)]
    for t in range(num_teams_plus_one):
        for g in range(num_genders):
            team_gender[t][g] = sum(matr_team_country_gender[t][c][g] for c in range(num_countries))
    A = sum(weights[t][c]*team_country[t][c] for t in range(1, num_teams_plus_one) for c in range(num_countries))
    B = (sum(team_country[t][c]*lambd1*team_country[t][c] for t in range(1, num_teams_plus_one) for c in range(num_countries)))
    C = (sum(team_gender[t][g]*lambd2*team_gender[t][g] for t in range(1, num_teams_plus_one) for g in range(num_genders)))
    return A + B + C
def initial_solution(country_capacities, team_demands, num_teams_plus_one, num_countries, num_genders):
    initial_matr_team_country_gender = [[[0 for k in range(num_genders)] for j in range(num_countries)] for i in range(num_teams_plus_one)]
    current_country, current_gender = 0, 0
    for i in range(1, num_teams_plus_one):
        remaining_demand_current_team = team_demands[i - 1]
        while remaining_demand_current_team > 0:            
            if remaining_demand_current_team >= country_capacities[current_country][current_gender]:
                
                initial_matr_team_country_gender[i][current_country][current_gender] += country_capacities[current_country][current_gender]
                remaining_demand_current_team -= country_capacities[current_country][current_gender]
                country_capacities[current_country][current_gender] = 0
                if current_gender == (num_genders - 1):
                    current_gender = 0
                    current_country += 1
                else:
                    current_gender += 1
            else:
                initial_matr_team_country_gender[i][current_country][current_gender] += remaining_demand_current_team
                country_capacities[current_country][current_gender] -= remaining_demand_current_team
                remaining_demand_current_team = 0
    for k in range(current_gender, num_genders):
        if current_country < num_countries:
            initial_matr_team_country_gender[0][current_country][k] += country_capacities[current_country][k]
    for j in range(current_country+1, num_countries):
        for k in range(num_genders):
            initial_matr_team_country_gender[0][j][k] += country_capacities[j][k]
    return initial_matr_team_country_gender



def match_entropy(res):
    #Function to calculate entropy of a matching for given clusters
    num_items,m=np.shape(res)
    ent=np.zeros((num_items,))
    for i in range(num_items):
        a=res[i,:]
        ent[i]=sp.entropy(a)
    return np.mean(ent)


if __name__ == '__main__':
    
    
    np.random.seed(1)     
    """Read UIUC numpy weights from data folder and implement
        1) Auxiliary graph Matching
        2) Diverse MIQP Matching"""

    #Load W1 and lab1            
    G = nx.DiGraph()
    W1=np.load('RevieweData/W1.npy')
    
    #Allocate labels to all reviewers
    lab1=np.load('RevieweData/lab1.npy')   
    
    #Allocate gender labels to every reviewer
    lab2=np.random.randint(0,2, [len(lab1),])    
    
    #Make two copies of each person, to meet the demands
    W1=np.concatenate((W1,W1))
    lab1=np.concatenate((lab1,lab1))      
    lab2=np.concatenate((lab2,lab2))
    
    num_countries = 1+max(lab1)     
    num_genders = 1+max(lab2)
    
    num_reviewers = 378 #authors
    num_papers = 13 #papers
    
    min_worker = 4 #minimum number of nodes that must match to each team
    min_country = 0 #  minimum number of nodes that must match to each country
    max_worker = num_reviewers # maximum number of nodes that can match to each worker
    
    
    Wmat = W1[0:num_reviewers,0:num_papers]    
    lab = lab1[0:num_reviewers] #every movie has a cluster label
    
    W2 = np.zeros((num_countries,num_papers))
    
   
    
    
    clus_capac = np.zeros(num_countries,)
    
    for j in range(num_countries):
        idx=np.where(lab == j)[0]
        clus_capac[j] = len(idx)
        for i in range(num_papers):
            #We average the weights of all people from a country to find the weight of country to team
            W2[j,i] = np.median(Wmat[idx,i])

    W = list(np.ravel(W2))
    
    #find the maximum number of people from each country/gender
    
    clus_gen_matr = np.zeros((num_countries, num_genders))
    for i in range(num_countries):
        find_clusidx=np.where(lab == i)[0]
        clus_gen = lab2[find_clusidx]
        for j in range(num_genders):
            clus_gen_matr[i,j] = len(np.where(clus_gen == j)[0])       
    
    lambd1, lambd2 = 1.0, 1.0
    num_teams_plus_one = num_papers+1
    num_teams = num_papers
    
    W2 = np.transpose(W2)
    wmatr_team_country_miqp = [list(i) for i in list(W2)]
    
    #Add zero weights to T0, which has unassigned people
    wmatr_team_country = [[0.0]*num_countries] + wmatr_team_country_miqp
    
    team_demands = [min_worker]*num_papers

    
    country_capacities = []
    c_capac = []
    for i in range(num_countries):
        country_capacities.append(list(clus_gen_matr[i,:]))
        c_capac.append(list(clus_gen_matr[i,:]))

    matr_team_country_gender = initial_solution(country_capacities, team_demands, num_teams_plus_one, num_countries, num_genders)
    find_graph_weighted_gender(matr_team_country_gender, wmatr_team_country, num_teams_plus_one,num_countries, num_genders,lambd1, lambd2)
    final_matr_tcg = worker_country_gender(matr_team_country_gender, wmatr_team_country, num_teams_plus_one,num_countries, num_genders,lambd1, lambd2)
    print ("Our algorithm objective function value: "+ objective_function(final_matr_tcg, num_teams_plus_one, num_countries, num_genders, wmatr_team_country, lambd1, lambd2))    
    mcg.tradeoff_matching_gender(num_teams, num_countries, 2, wmatr_team_country_miqp, team_demands,c_capac, 1.0, 1.0, 60)
