import numpy as np

def get_3d_config(A,x0,nt,z_index):
    
    A, x0, nt, z_index = np.array(A), np.array(x0), np.array(nt), np.array(z_index)
    
    n_joints = (A.sum(-1)>0).sum()
    A = A[:n_joints,:][:,:n_joints]
    x0 = x0[:n_joints]
    nt = nt[:n_joints]
    n_links = int(A.sum()/2)

    l1,l2 = np.where(np.triu(A))
    
    linkages = []
    
    max_len = 0
    min_len = float(np.inf)

    for j in range(n_links):
        length= np.linalg.norm(x0[l1[j]]-x0[l2[j]])
        if length>max_len:
            max_len = float(length)
        if length<min_len:
            min_len = float(length)
    
    scale_min = 0.25/min_len
    scale_target = 1.0/max_len
    scale = max(scale_min,scale_target)
    
    x0 = x0*scale

    for j in range(n_links):
        length= np.linalg.norm(x0[l1[j]]-x0[l2[j]])
        angle = np.arctan2(x0[l2[j]][1]-x0[l1[j]][1],x0[l2[j]][0]-x0[l1[j]][0])
        linkages.append([length,0.1,0.05,0.03, angle, x0[l1[j]].tolist()+[float(z_index[j])*0.05]])
    
    joints_max_z = np.zeros(x0.shape[0])
    joints_min_z = np.zeros(x0.shape[0]) + np.inf
    
    for i in range(n_links):
        joints_max_z[l1[i]] = max(joints_max_z[l1[i]],z_index[i])
        joints_max_z[l2[i]] = max(joints_max_z[l2[i]],z_index[i])
        joints_min_z[l1[i]] = min(joints_min_z[l1[i]],z_index[i])
        joints_min_z[l2[i]] = min(joints_min_z[l2[i]],z_index[i])

    joints_max_z = joints_max_z*0.05
    joints_min_z = joints_min_z*0.05

    joints = []
    for i in range(x0.shape[0]):
        if nt[i] == 1:
            for j in np.where(l1==i)[0]:
                joints.append(x0[i].tolist()+[0.05,z_index[j]*0.05,1])
            for j in np.where(l2==i)[0]:
                joints.append(x0[i].tolist()+[0.05,z_index[j]*0.05,1])
        else:
            joints.append(x0[i].tolist()+[float(joints_max_z[i]-joints_min_z[i])+0.05,float(joints_min_z[i]+joints_max_z[i])/2,0])

    return [linkages, joints], joints_max_z, scale

def get_animated_3d_config(A,x0,nt,z_index,sol, highlights = [-1]):
    A, x0, nt, z_index, sol = np.array(A), np.array(x0), np.array(nt), np.array(z_index), np.array(sol)
    configs = []
    for i in range(sol.shape[1]):
        c,z,s = get_3d_config(A,sol[:,i,:],nt,z_index)
        configs.append(c)
    
    if len(highlights) > 1:
        highligh_curve = []
        for i in highlights:
            highligh_curve.append(np.pad(sol[i,:,:]*s,[[0,0],[0,1]],constant_values=z[i]+0.025))
        highligh_curve = np.array(highligh_curve)
    else:
        highligh_curve = np.pad(sol[highlights[0],:,:]*s,[[0,0],[0,1]],constant_values=z[-1]+0.025)

    return configs, highligh_curve.tolist()

def create_3d_html(A,x0,nt,z_index,sol, template_path='./static/animation.htm', save_path='./static/animated.html', highlights = [-1]):
    
    res,hc = get_animated_3d_config(A,x0,nt,z_index,sol,highlights=highlights)
    js_var = 'window.res = ' + str(res) + ';\n'
    js_var += 'window.hc = ' + str(hc) + ';'
    js_var += 'window.multi_high = ' + str(int(len(highlights)>1)) + ';'
    
    with open(template_path, 'r') as file:
        filedata = file.read()
        
    filedata = filedata.replace('{res}',js_var)
    
    with open(save_path, 'w') as file:
        file.write(filedata)