"""
Code for IJCAI 2017 paper titled "Diverse Weighted Bipartite b-Matching"
Ahmed, Faez, John P. Dickerson, and Mark Fuge
"""

from gurobipy import *
import numpy as np
import time
from scipy.sparse import csr_matrix
import collections
import scipy.stats as sp


def edge_index(idx,num_left,num_right):
    #This function takes a node id from bi-partite graph and returns id  of edges connected to it
    if(idx<num_left):
        vec=np.arange(idx*num_right,(idx+1)*num_right)
    else:
        paper_id=idx-num_left
        vec=np.arange(paper_id,num_left*num_right,num_right)
    return list(vec)

def linkmatr(num_left,num_right):
    #Creates link matrix A used to define degree constraints for all nodes
    num_nodes=num_left+num_right
    str1=[1]*num_right
    str2=[0]*num_right
    A=[None]*(num_nodes)
    for i in range(num_left):
        A[i]=str2*num_left
        #print A[i]
        idx=num_right*i
        A[i][idx:idx+num_right]=str1
    for j in range(num_right):
        A[num_left+j]=str2*num_left
        idx=[j+num_right*l for l in range(num_left)]
        for k in range(num_left):
            A[num_left+j][idx[k]]=1      
    return A

def sparse_block_wt(W,num_left,num_right,lab,rowflag):
  
    #calculate weight matrix D for diversity
    #if rowflag is 0, return only the sparse form.  
    total_vars=num_left*num_right
    num_nodes=num_left+num_right
    row=[]
    col=[]
    val=[]
    num_clusters=1+max(lab)
    
    #cluster label for every edge
    labels=np.zeros((total_vars,))   
    for i in range(len(lab)):
        labels[i*num_right:(i+1)*num_right]=np.tile(lab[i],num_right)  
    
    #row, column and value for block diagonal matrix    
    for p in range(num_left,num_nodes):
        ed=edge_index(p,num_left,num_right)
        for i in range(num_clusters):                
            idx=np.intersect1d(np.where(labels==i)[0],ed)
            num_mem=len(idx)
            for j in range(num_mem):
                for k in range(num_mem):                    
                    row.append(idx[j])
                    col.append(idx[k])
                    val.append(W[idx[j]]*W[idx[k]])
                     
    Ws=csr_matrix((val, (row, col)), shape=(total_vars, total_vars))

    if(rowflag==0):
        return Ws
    else:
        return Ws, row, col


def match_entropy(res,lab,num_right):
    #Function to calculate entropy of a matching for given clusters
    
    #Input: res MxN matching matrix 
    #lab: cluster labels for M
    #num_right is N (Number of nodes on Right side)
    #Output: Entropy of each node and average entropy
    ent=np.zeros((num_right,))
    for i in range(num_right):
        a=np.where(res[:,i])
        cls=np.array(lab[a[0]])
        counter=collections.Counter(cls)
        ent[i]=sp.entropy(counter.values())
    return ent, np.mean(ent)
    
    
def set2matr(sset,num_left,num_right):
    #Takes a list of edges and returns a matrix with 1's and 0's
    res=np.zeros((num_left,num_right))
    for i in range(len(sset)):
        k=sset[i]
        ii=int(k/num_right)
        jj=k%num_right
        res[ii,jj]=1
    return res
    
def matr2set(res):
    #Takes a matrix with 1's and 0's and returns list of edges
    return list(np.where(np.ravel(res)==1))
    
def sol_fitness(sset,W,row,col):
    #Returns Wx and x'Bx when set indices are provided
    f1=np.sum(W[sset])
    f2=0
    for i in range(len(row)):
        if((row[i] in sset) & (col[i] in sset)):
            f2=f2+W[row[i]]*W[col[i]]
    return f1,f2
    
def node2from(node_id,num_left,num_right):
    #Returns the left and right sequence id
    node_left=int(node_id/num_right)
    node_right=num_left+int(node_id%num_right)
    return node_left,node_right
    
def author_matching(num_left,num_right, W,lda,uda,ldp,udp):
    ##D and W are list
    try:
        # Create a new model
        m = Model("mip1")
        #m.setParam("OutputFlag", 0);
        #m.setParam("MIPFocus", 1)
        

        total_nodes = num_left+num_right
        total_vars = num_left*num_right
        
        if((num_left*lda> num_right*udp) or (num_right*ldp>num_left*uda)):
            print 'Infeasible Problem'
            return
        
        #Maximum Number of authors matched to node paper
        Dmax=list(udp*np.ones((total_nodes,)))
        
        #Minimum Number of authors matched to a paper
        Dmin=list(ldp*np.ones((total_nodes,)))
        
        #Minimum Number of papers matched to an author
        Dmina=list(lda*np.ones((total_nodes,)))
        
        #Maximum Number of papers matched to author
        Dmaxa=list(uda*np.ones((total_nodes,)))

        
        A=linkmatr(num_left,num_right)
        x = {}
        for j in range(total_vars):
          x[j] = m.addVar(vtype=GRB.BINARY, name="x"+str(j))

        #Set objective
        m.setObjective((quicksum(W[i]*x[i] for i in range(total_vars))), GRB.MINIMIZE)
        
        #constraint on paper cardinality
        for i in range(num_left,total_nodes):
            m.addConstr(quicksum(A[i][j]*x[j] for j in range(total_vars))<=Dmax[i])
            m.addConstr(quicksum(A[i][j]*x[j] for j in range(total_vars))>=Dmin[i])
            
        #constraint on authors
        for i in range(num_left):
            m.addConstr(quicksum(A[i][j]*x[j] for j in range(total_vars))<=Dmaxa[i])
            m.addConstr(quicksum(A[i][j]*x[j] for j in range(total_vars))>=Dmina[i])    

        #m.write("lp.mps")    
        # Optimize
        m.optimize()   
        
        res=np.zeros((num_left,num_right))
        for i in range(num_left):
            for j in range(num_right):
                idx=num_right*i+j
                res[i,j]=m.getVars()[idx].x

        return res        

    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
        

def diverse_matching(num_left,num_right, W,lab,lda,uda,ldp,udp,row,col,tim):
    #Diverse matching MIQP    
    try:
        # Create a new model
        m = Model("mip1")
        m.setParam("NodefileStart", 0.5)
        #m.setParam("threads", 24)
        #m.setParam("MIPGap", 0.05)
        m.setParam("MIPFocus", 3)
        
        #Set maximum running time
        m.Params.timelimit = tim
        #m.setParam("OutputFlag", 0)

        total_nodes = num_left+num_right
        total_vars = num_left*num_right
        
        #Maximum Number of authors matched to node paper
        Dmax=list(udp*np.ones((total_nodes,)))
        
        #Minimum Number of authors matched to a paper
        Dmin=list(ldp*np.ones((total_nodes,)))
        
        #Minimum Number of papers matched to an author
        Dmina=list(lda*np.ones((total_nodes,)))
        
        #Maximum Number of papers matched to author
        Dmaxa=list(uda*np.ones((total_nodes,)))

        A=linkmatr(num_left,num_right)
        
        start_time = time.time()
        x = {}
        for j in range(total_vars):
          x[j] = m.addVar(vtype=GRB.BINARY, name="x"+str(j))

        #Set objective
        m.setObjective((quicksum(x[i]*W[i]*W[j]*x[j] for i,j in zip(row,col))), GRB.MINIMIZE)        
        #constraint on paper cardinality
        for i in range(num_left,total_nodes):
            m.addConstr(quicksum(A[i][j]*x[j] for j in range(total_vars))<=Dmax[i])
            m.addConstr(quicksum(A[i][j]*x[j] for j in range(total_vars))>=Dmin[i])
            
        #constraint on authors
        for i in range(num_left):
            m.addConstr(quicksum(A[i][j]*x[j] for j in range(total_vars))<=Dmaxa[i])
            m.addConstr(quicksum(A[i][j]*x[j] for j in range(total_vars))>=Dmina[i])    
              
        print "Optimizing Now"   
        
        #m.write("qp.mps")
        # Optimize
        m.optimize()   
        print("--- %s seconds ---" % (time.time() - start_time))
        
        
        res=np.zeros((num_left,num_right))
        for i in range(num_left):
            for j in range(num_right):
                idx=num_right*i+j 
                res[i,j]=m.getVars()[idx].x

            
        return res

    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
        
        
def safe_greedy_diverse(num_left,num_right, Ws,lda,uda,ldp,udp,row,col): 
    
    #Greedy code which gradually increases lower constraint for each node    
    #initialize number of authors and papers, total nodes and total edges
    
    num_nodes=num_left+num_right
    #Generate the link matrix A
    #A=linkmatr(num_left,num_right)   
    
    if((num_left*lda> num_right*udp) or (num_right*ldp>num_left*uda)):
        print 'Infeasible Problem'
        return None  
        
    start_time = time.time()
    
    #Provide any ordering to satisfy lower bounds
    order=np.arange(num_nodes)[::-1]   
    #np.random.shuffle(order)

    #Takes an order and finds a feasible set satisfying lower constraints
    #curr_degree is current degree of a node as iterations proceed    
    curr_degree=np.zeros((num_left+num_right,)).astype('int64')
    sset=[]
    
    #Dup is upper degree constraint
    Dup=list(uda*np.ones((num_left,)).astype('int64'))+list(udp*np.ones((num_right,)).astype('int64'))

        
    big_number=10*np.sum(W)
    if(lda>ldp):
        lowerbnd=lda;
    else:
        lowerbnd=ldp;
        
    ldacnt=0;
    ldpcnt=0;
    start_time=time.time()
    
    for loopi in range(lowerbnd):
        if(ldacnt<lda):
            ldacnt=ldacnt+1;
        else:
            ldacnt=lda;
        if(ldpcnt<ldp):
            ldpcnt=ldpcnt+1;
        else:
            ldpcnt=ldp;     
        #D is current lower degree constraint 
        D=list(ldacnt*np.ones((num_left,)).astype('int64'))+list(ldpcnt*np.ones((num_right,)).astype('int64'))
        
        for k in range(num_nodes):
            #Node to satisfy constraint
            node_id=order[k]                
            #l1 is list of all edges from node and vec is all edges not matched yet
            l1=edge_index(node_id,num_left,num_right)
            vec = [x for x in l1 if x not in sset]
            edges_needed=D[node_id] +len(vec) -len(l1)
            
            for n in range(edges_needed):

                #num_ed is number of edges of this node which are not matched yet
                num_ed=len(vec)
                
                obj=big_number*np.ones((num_ed,))
                for ii in range(num_ed):
                    #l is an edge not matched yet
                    l=vec[ii]
                    
                    #add edge to the set and calculate objective and constraints
                    tempset=sset+[l]
                    node_to,node_from=node2from(l,num_left,num_right)

                    if((curr_degree[node_to]<Dup[node_to]) & (curr_degree[node_from]<Dup[node_from])):
                        obj[ii] = 2*(np.sum(Ws[l,i] for i in tempset))-Ws[l,l]
                        #If any node is below minimum degree quota prefer it by penalizing all others
                        if((node_id==node_to) & (curr_degree[node_from]>=D[node_from])):
                            obj[ii]=obj[ii]+0.5*big_number
                        elif((curr_degree[node_to]>=D[node_to]) & (node_id==node_from)):
                            obj[ii]=obj[ii]+0.5*big_number
             
                if(np.min(obj)!=big_number):
                    idx= np.argmin(obj)
                    sset.append(vec[idx])
                    node_to,node_from=node2from(vec[idx],num_left,num_right)                                    
                    curr_degree[node_from]=curr_degree[node_from]+1
                    curr_degree[node_to]=curr_degree[node_to]+1
                    vec.pop(idx)
        

    print("--- %s seconds ---" % (time.time() - start_time))
    return sset

if __name__ == "__main__":
    
    np.random.seed(4)

    #UIUC dataset 73 papers and 189 reviewers. M=189, N=73  
    #Each reviewer reviews atleast one and maximum 10 papers. Each paper gets 3 reviewers.

    try:
        W1
        lab1
    except NameError:
        W1=np.load('W1.npy')
        lab1=np.load('lab1.npy')
        
    num_clusters=5   
    num_right=73 #73 papers
    num_left=189 #189 reviewers
    ldp=3   # Minimum paper cardinality
    udp=3   # Maximum paper cardinality
    uda=10  # maximum papers one reviewer will review
    lda=1   # minimum papers every reviewer has to review
    total_nodes = num_left+num_right
    total_vars = num_left*num_right  
    
    W=list(np.ravel(W1))    
    lab=lab1[0:num_left]

    Ws, row, col= sparse_block_wt(W,num_left,num_right,lab,1)
    
    
    
    #WBM weighed matching
    start_time = time.time()
    res_direct=np.round(author_matching(num_left,num_right, W,lda,uda,ldp,udp))
    wtime=time.time()-start_time
    #Calculate entropy of direct matching
    ent_direct, avg_ent_direct=match_entropy(res_direct,lab,num_right) 
    
    #D-WBM diverse matching with 360 seconds timelimit
    start_time = time.time()
    res= np.round(diverse_matching(num_left,num_right, W,lab,lda,uda,ldp,udp,row,col,360))  
    ent, avg_ent=match_entropy(res,lab,num_right) 
    dtime=time.time()-start_time
    
    #GD-WBM Greedy matching
    start_time = time.time()
    greedy_set= safe_greedy_diverse(num_left,num_right,Ws,lda,uda,ldp,udp,row,col) 
    res_greedy=set2matr(greedy_set,num_left,num_right)
    ent_greedy, avg_ent_greedy=match_entropy(res_greedy,lab,num_right) 
    gtime=time.time()-start_time
    
    #Direct Matching and Diverse Matching Fitness for the three methods
    f1=sol_fitness(matr2set(res)[0],np.array(W),row,col)
    f2=sol_fitness(greedy_set,np.array(W),row,col)
    f3=sol_fitness(matr2set(res_direct)[0],np.array(W),row,col)
    
    
    print "DP-WBM Fitness"+str(f1)+"\nEntropy "+str(avg_ent)+" e= "+str(np.sum(res))
    print "GDP-WBM Fitness"+str(f2)+"\nEntropy "+str(avg_ent_greedy)+" e= "+str(np.sum(res_greedy))
    print "WBM Fitness"+str(f3)+"\nEntropy "+str(avg_ent_direct)+" e= "+str(np.sum(res_direct))
    
    
    print "Price of Diversity DP " +str(f3[0]/f1[0])+" Entropy Gain " +str(avg_ent/avg_ent_direct)
    print "Price of Diversity GDP " +str(f3[0]/f2[0])+" Entropy Gain " +str(avg_ent_greedy/avg_ent_direct)           
        
