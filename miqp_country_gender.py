from gurobipy import *
import sys
import time
def tradeoff_matching_gender( num_teams, num_countries, num_genders, weights,demands, capacity, lambd1, lambd2,tim):
    isgender = 1
    print ("before the constraints")
    #Diverse matching MIQP    
    try:
        # Create a new model
        m = Model("mip1")
        m.setParam("NodefileStart", 0.5)
        #m.setParam("threads", 1)
        m.Params.timelimit = tim       
        start_time = time.time()
        
        team_country = [[m.addVar(vtype=GRB.INTEGER, name="team_country"+str(t*num_countries + c)) for c in range(num_countries)] for t in range(num_teams)]
        
        if(isgender==1):
            team_country_gender = [[[m.addVar(vtype=GRB.INTEGER, name="team_country_gender"+str(t*(num_countries*num_genders) + c*(num_genders)+ g))for g in range(num_genders)] for c in range(num_countries)] for t in range(num_teams)]
            team_gender = [[m.addVar(vtype=GRB.INTEGER, name="team_gender"+str(t*num_genders + g)) for g in range(num_genders)] for t in range(num_teams)]
            m.setObjective((quicksum(weights[t][c]*team_country[t][c] for t in range(num_teams) for c in range(num_countries)))+(quicksum(team_country[t][c]*lambd1*team_country[t][c] for t in range(num_teams) for c in range(num_countries))) + \
                (quicksum(team_gender[t][g]*lambd2*team_gender[t][g] for t in range(num_teams) for g in range(num_genders))), GRB.MINIMIZE) 
            #constraints
            for t in range(num_teams):
                for c in range(num_countries):
                    for g in range(num_genders):
                        m.addConstr(team_country_gender[t][c][g]>=0)   
            for c in range(num_countries):
                for t in range(num_teams):
                    m.addConstr(quicksum(team_country_gender[t][c][g] for g in range(num_genders)) == team_country[t][c])
            for t in range(num_teams):
                for g in range(num_genders):
                    m.addConstr(quicksum(team_country_gender[t][c][g] for c in range(num_countries)) == team_gender[t][g])
            for t in range(num_teams):
                m.addConstr(quicksum(team_country_gender[t][c][g] for c in range(num_countries) for g in range(num_genders)) == demands[t])
            for c in range(num_countries):
                for g in range(num_genders):
                    m.addConstr(quicksum(team_country_gender[t][c][g] for t in range(num_teams)) <= capacity[c][g])
        print ("Optimizing Now")
        # Optimize
        m.optimize()   
        print("--- %s seconds ---" % (time.time() - start_time))    
                
        status = m.status  
    
        if status == GRB.Status.UNBOUNDED:
            print('The model cannot be solved because it is unbounded')
        elif status == GRB.Status.OPTIMAL:
            print('The optimal objective is %g' % m.objVal)
            for v in m.getVars():
                print(v.varName, v.x)
        elif status != GRB.Status.INF_OR_UNBD and status != GRB.Status.INFEASIBLE:
            print('Optimization was stopped with status %d' % status)
            
        return m
    
    except GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))
    
    except AttributeError:
        print('Encountered an attribute error')
        
#def main():
#    print ("check")
    #weights = [[1,2,6], [3,4,8]]
    #demands = [12, 11]
    #capacity = [[10,3], [15,4], [3, 6]]
    #tradeoff_matching_gender(2, 3, 2, weights, demands, capacity, 1.0, 1.0, 60)
#main()