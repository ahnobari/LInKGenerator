import torch

def uniformize(curves: torch.tensor, n: int = 200) -> torch.tensor:
    with torch.no_grad():
        l = torch.cumsum(torch.nn.functional.pad(torch.norm(curves[:,1:,:] - curves[:,:-1,:],dim=-1),[1,0,0,0]),-1)
        l = l/l[:,-1].unsqueeze(-1)
        
        sampling = torch.linspace(0,1,n).to(l.device).unsqueeze(0).tile([l.shape[0],1])
        end_is = torch.searchsorted(l,sampling)[:,1:]
        end_ids = end_is.unsqueeze(-1).tile([1,1,2])
        
        l_end = torch.gather(l,1,end_is)
        l_start = torch.gather(l,1,end_is-1)
        ws = (l_end - sampling[:,1:])/(l_end-l_start)
    
    end_gather = torch.gather(curves,1,end_ids)
    start_gather = torch.gather(curves,1,end_ids-1)
    
    uniform_curves = torch.cat([curves[:,0:1,:],(end_gather - (end_gather-start_gather)*ws.unsqueeze(-1))],1)

    return uniform_curves