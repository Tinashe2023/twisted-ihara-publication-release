"""Draw the exact Tanner graphs and their nonisomorphism certificates.
Run: python make_tanner_figure.py
Dependencies: numpy, matplotlib, networkx
Outputs: tanner_graphs.png and tanner_graphs.svg
"""
from pathlib import Path
from itertools import combinations
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

H1 = np.array([
 [1,0,0,0,1,1,0,0],
 [0,1,0,0,0,1,0,1],
 [0,1,0,0,1,0,1,0],
 [0,0,0,1,0,0,1,1],
 [0,0,1,0,1,0,1,0],
 [1,0,1,1,0,0,0,0],
 [1,0,1,0,0,1,0,0],
 [0,1,0,1,0,0,0,1]], dtype=int)
H2 = np.array([
 [1,0,0,1,1,0,0,0],
 [0,0,0,0,1,1,0,1],
 [0,0,1,1,0,0,1,0],
 [1,1,0,0,1,0,0,0],
 [0,1,1,0,0,0,1,0],
 [0,0,0,1,0,0,1,1],
 [1,0,0,0,0,1,0,1],
 [0,1,1,0,0,1,0,0]], dtype=int)

plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                     "svg.fonttype":"none"})

def build_graph(H):
    G = nx.Graph()
    G.add_nodes_from(range(16))  # 0..7: checks; 8..15: variables
    G.add_edges_from((c,8+v) for c,v in zip(*np.nonzero(H)))
    assert nx.is_connected(G)
    assert all(d == 3 for _,d in G.degree())
    counts = {w: sum(len(set(G[a]) & set(G[b]))-1
                    for a,b in combinations(G[w],2)) for w in G}
    S = {w for w in G if counts[w] == 2}
    assert all(counts[w] == 1 for w in G if w not in S)
    return G,S

def draw(ax,G,S,pos,labels=True,background=True):
    ordinary=[(a,b) for a,b in G.edges() if not (a in S and b in S)]
    special=[(a,b) for a,b in G.edges() if a in S and b in S]
    if background:
        nx.draw_networkx_edges(G,pos,edgelist=ordinary,ax=ax,
                               edge_color="#aeb7c0",width=1.1,alpha=.8)
    nx.draw_networkx_edges(G,pos,edgelist=special,ax=ax,
                           edge_color="#b13c20",width=2.7)
    for nodes,shape in [(range(8),"s"),(range(8,16),"o")]:
        nodes=[w for w in nodes if w in G]
        nx.draw_networkx_nodes(G,pos,nodelist=nodes,node_shape=shape,
            node_size=310 if background else 360,
            node_color=["#f8ce91" if w in S else "white" for w in nodes],
            edgecolors="#20252a",linewidths=1.1,ax=ax)
    if labels:
        names={w:rf"$c_{{{w}}}$" if w<8 else rf"$v_{{{w-8}}}$" for w in G}
        nx.draw_networkx_labels(G,pos,labels=names,font_size=9,ax=ax)
    ax.set_aspect("equal",adjustable="box")
    ax.axis("off")

fig,axs=plt.subplots(2,2,figsize=(10,7.5),
                     gridspec_kw={"height_ratios":[3.2,1.2]})
for j,H in enumerate([H1,H2]):
    G,S=build_graph(H)
    expected=[{6,7,8,15},{2,4,10,14}][j]
    assert S == expected
    induced=G.subgraph(S).copy()
    assert induced.number_of_edges() == [2,4][j]
    # A fixed bipartite layout preserves the manuscript's index order.
    pos={c:(0,7-c) for c in range(8)}
    pos.update({8+v:(4.6,7-v) for v in range(8)})
    draw(axs[0,j],G,S,pos)
    axs[0,j].set_title(rf"({'ab'[j]}) $X_{j+1}$",fontsize=12,pad=12)
    axs[0,j].set_xlim(-.65,5.25)
    axs[0,j].set_ylim(-.55,7.55)
    if j==0:
        small={6:(0,1),8:(0,0),7:(2,1),15:(2,0)}
        subtitle=r"$X_1[S(X_1)]$: two disjoint edges"
    else:
        small={2:(0,1),4:(2,1),10:(0,0),14:(2,0)}
        subtitle=r"$X_2[S(X_2)]$: a four-cycle"
    draw(axs[1,j],induced,S,small,background=False)
    axs[1,j].set_xlim(-.6,2.6)
    axs[1,j].set_ylim(-.5,1.5)
    axs[1,j].set_title(subtitle,fontsize=10,pad=4)
    print(f"X{j+1}: highlighted vertices {sorted(S)}; "
          f"induced edges {sorted(induced.edges())}")

legend=[
 Line2D([],[],marker="s",linestyle="none",markerfacecolor="white",
        markeredgecolor="#20252a",label="Check vertex"),
 Line2D([],[],marker="o",linestyle="none",markerfacecolor="white",
        markeredgecolor="#20252a",label="Variable vertex"),
 Line2D([],[],marker="o",linestyle="none",markerfacecolor="#f8ce91",
        markeredgecolor="#20252a",label="Vertex on exactly two four-cycles"),
 Line2D([],[],color="#b13c20",linewidth=2.7,label="Edge induced by highlighted vertices")]
fig.legend(handles=legend,loc="lower center",ncol=2,frameon=False,
           fontsize=9,bbox_to_anchor=(.5,.01))
fig.subplots_adjust(left=.06,right=.94,top=.94,bottom=.14,hspace=.23,wspace=.25)
out=Path(__file__).resolve().parent
fig.savefig(out/"tanner_graphs.png",dpi=400,bbox_inches="tight",facecolor="white")
fig.savefig(out/"tanner_graphs.svg",bbox_inches="tight",facecolor="white")
plt.close(fig)
print(f"Saved figure files in {out}")
