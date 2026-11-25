import trimesh,numpy as np,sys
from PIL import Image
from scipy.ndimage import map_coordinates
flip=int(sys.argv[1]); out=sys.argv[2]
m=trimesh.load("mv_white.glb",force="mesh"); m.merge_vertices(); m.update_faces(m.nondegenerate_faces()); m.remove_unreferenced_vertices()
m=m.simplify_quadric_decimation(face_count=150000)
v=m.vertices; mn,mx=v.min(0),v.max(0); vn=m.vertex_normals
def load(f): return np.array(Image.open(f).convert("RGBA")).astype(float)
def samp(img,u,w):
    H,W=img.shape[:2]; uu=np.clip(u,0,1)*(W-1); ww=np.clip(w,0,1)*(H-1)
    return np.stack([map_coordinates(img[...,k],[ww,uu],order=1,mode='nearest') for k in range(4)],1)
y=(mx[1]-v[:,1])/(mx[1]-mn[1])
x=(v[:,0]-mn[0])/(mx[0]-mn[0]); z=(v[:,2]-mn[2])/(mx[2]-mn[2])
views={"mv_front.png":(x,vn[:,2]),"mv_back.png":(1-x,-vn[:,2]),
 "mv_left.png":((z if not flip else 1-z),vn[:,0]*(-1 if not flip else 1)),
 "mv_right.png":((1-z if not flip else z),vn[:,0]*(1 if not flip else -1))}
acc=np.zeros((len(v),3));ws=np.zeros(len(v))
for f,(u,n) in views.items():
    c=samp(load(f),u,y); side=("front" not in f and "back" not in f)
    wt=np.clip(n,0,1)**3*(c[:,3]>200)
    if side: wt=np.clip(n-0.55,0,1)**2*(c[:,3]>200)*np.clip(1-3*np.maximum(np.clip(vn[:,2],0,1),np.clip(-vn[:,2],0,1)),0,1)
    else: wt=np.clip(n-0.1,0,1)**2*(c[:,3]>200)
    acc+=c[:,:3]*wt[:,None]; ws+=wt
base=np.array([80,70,65.])
col=np.where(ws[:,None]>1e-3,acc/np.maximum(ws,1e-6)[:,None],base)
vc=np.clip(col,0,255).astype(np.uint8)
m.visual=trimesh.visual.ColorVisuals(m,vertex_colors=np.c_[vc,np.full(len(vc),255,np.uint8)]); m.export(out)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
fig,axs=plt.subplots(1,6,figsize=(18,4.5)); f=m.faces
for ax,ang in zip(axs,[0,60,90,180,270,300]):
    a=np.radians(ang); R=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    P=(v-(mn+mx)/2)@R.T; d=P[:,2]
    fn=np.cross(P[f][:,1]-P[f][:,0],P[f][:,2]-P[f][:,0]); fn/=np.linalg.norm(fn,axis=1,keepdims=True)+1e-9
    o=np.argsort(d[f].mean(1)); fc=vc[f].mean(1)/255.; sh=(0.6+0.4*np.abs(fn[:,2]))[:,None]
    ax.add_collection(PolyCollection(P[f][:,:,:2][o],facecolors=np.clip(fc*sh,0,1)[o],edgecolors="none",antialiaseds=False))
    r=(mx-mn).max()/2; ax.set_xlim(-r,r); ax.set_ylim(-r*1.05,r*1.05); ax.set_aspect('equal'); ax.axis('off')
plt.tight_layout(); plt.savefig(out[:-4]+".png",dpi=60,facecolor='white')
