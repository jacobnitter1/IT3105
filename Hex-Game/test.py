def server_state_to_NN_state(srv_state):
    NN_state=[]
    for i in range(1,len(srv_state)):
        #print(i)
        if srv_state[i] == 0:
            NN_state.append(0)
            NN_state.append(0)
        elif srv_state[i] == 1:
            NN_state.append(1)
            NN_state.append(0)
        elif srv_state[i] == 2:
            NN_state.append(0)
            NN_state.append(1)
    #APPEND PLAYER AT END
    if srv_state[0] == 2:
        NN_state.append(0)
        NN_state.append(1)
    else:
        NN_state.append(1)
        NN_state.append(0)
    return NN_state

def NN_state_to_server_state(NN_state):
    srv_state=[]
    if NN_state[-2] == 0:
        if NN_state[-1] == 1:
            srv_state.append(2)
    else:
        srv_state.append(1)
    for i in range(0,len(NN_state)-2,2):
        #print(i)
        if NN_state[i] == 0:
            if NN_state[i+1] == 1:
                srv_state.append(2)
            else:
                srv_state.append(0)
        else:
            srv_state.append(1)


    return srv_state
print("Server state : ",[2,0,0,0,2,0,0,1,0,2,0,0,0,1,0,2,0])
a =server_state_to_NN_state([2,0,0,0,2,0,0,1,0,2,0,0,0,1,0,2,0])
print("NN state : ",a)
print("Server state : ",NN_state_to_server_state(a))
