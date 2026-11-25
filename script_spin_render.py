import trimesh,numpy as np
from PIL import Image
m=trimesh.load("man_col_final.glb",force="mesh")
v=m.vertices; vc=np.asarray(m.visual.vertex_colors)[:,:3].astype(float)
n=m.vertex_normals
mn,mx=v.min(0),v.max(0); c=(mn+mx)/2; r=(mx-mn).max()/2
Wd,Ht=720,960; S=Ht/(r*2*1.35)
bg=np.array([233,228,218],float)
def ease(t): return t*t*(3-2*t)
def render(ang):
    a=np.radians(ang); R=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    P=(v-c)@R.T; Nn=n@R.T
    depth=P[:,2]              # camera at -z: smaller z = nearer
    order=np.argsort(-depth)  # far first
    x=(Wd/2 - P[:,0]*S).astype(int); y=(Ht/2 - P[:,1]*S).astype(int)
    shade=0.55+0.45*np.abs(Nn[:,2]); col=np.clip(vc*shade[:,None],0,255)
    img=np.tile(bg,(Ht,Wd,1))
    xs=[];ys=[];cs=[];ds=[]
    for dx in range(-3,4):
        for dy in range(-3,4):
            xs.append(np.clip(x+dx,0,Wd-1)); ys.append(np.clip(y+dy,0,Ht-1)); cs.append(col); ds.append(depth)
    X=np.concatenate(xs);Y=np.concatenate(ys);C=np.concatenate(cs);D=np.concatenate(ds)
    o=np.argsort(-D)
    img[Y[o],X[o]]=C[o]
    return Image.fromarray(img.astype(np.uint8))
N=28
for i in range(N+1):
    render(360*ease(i/N)).save(f"fr/s{i:03d}.png")
