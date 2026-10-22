#coding:utf-8
import torch
from scipy.stats import beta
import numpy as np
from torch import nn
import torch.nn.functional as F
from openTSNE import TSNE
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.cbook as cbook
from matplotlib.patches import Circle
from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)
from matplotlib.cbook import get_sample_data
import matplotlib.image as mimage
import pickle
import os
import cv2
point_shape = ['.', 'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd']
point_color = ['b', 'g', 'r', 'm', 'k', 'y', 'w', 'c']

def tsne_plot(feat, name):
    length= 1000
    fig = plt.figure()
    feats_1 = feat[:length]
    feats_2 = feat[l1:l1+length]
    feats_ = torch.cat([feats_1, feats_2],dim=0).view(2*length,-1).data.numpy()
    
    x_2d = TSNE(perplexity=10).fit(feats_)
    x_1 = x_2d[:length, 0]
    y_1 = x_2d[:length, 1]
    x_2 = x_2d[length:length*2, 0]
    y_2 = x_2d[length:length*2, 1]
    size = 1
    plt.scatter(x_1, y_1, marker='o', s=size, label="market")
    plt.scatter(x_2, y_2, marker='^', s=size, label="duke")
    #plt.scatter(x_3, y_3, marker='s', s=size, label="MSMT")
    #plt.scatter(x_4, y_4, marker='D', s=size, label="Cuhk03")
    plt.savefig(name)
    plt.close(0) 






def relabel(source_ids):
    own_sources = list(set(source_ids.tolist()))
    source_ids_ = []
    for s in source_ids:
        source_ids_.append(own_sources.index(s))
    source_ids_ = np.array(source_ids_)
    return source_ids_




def tsne_drawer(source_ids, attn_maps, extracted_features, pids, im_paths):
    #fig = plt.figure()
    for hk in range(5):
        attn_maps_ = attn_maps[:,hk,:]
        extracted_features_ = extracted_features[:,:,hk]
        fig, ax = plt.subplots(figsize=(20, 20))
        source_ids_ = relabel(source_ids)
        for i in range(len(source_ids)):
            index = np.where(source_ids_==i)[0]
            feat = extracted_features_[index]
            if len(feat.shape)==4:
                N, C, H, W = feat.shape
                feat = feat.view(N, C, -1).mean(2)
            if i==0:
                feats_ = feat
            else:
                feats_ = torch.cat([feats_, feat], dim=0)

        x_2d = TSNE(perplexity=30).fit(feats_)
        size = 80
        
        


        point_scatter(source_ids_, pids, x_2d, ax)
        plt.savefig('./tsne/'+str(hk)+'_point.jpg')

        imscatter(x_2d[:,0], x_2d[:,1], im_paths, attn_maps_, ax, 1)    

            
        #plt.show()
        plt.savefig('./tsne/'+str(hk)+'.jpg')
        imscatter(x_2d[:,0], x_2d[:,1], im_paths, attn_maps_, ax, 1, True)   
        plt.savefig('./tsne/'+str(hk)+'_o.jpg')
        print('done3')
        plt.close(0) 
    partitions = {'source_ids': source_ids,
              'attn_maps': attn_maps,
              'extracted_features': extracted_features,
              'pids': pids,
              'im_paths': im_paths}

    with open('./tsne/partitions.pkl', 'wb') as f:
        pickle.dump(partitions, f, protocol=2)    
    return 
    
def point_scatter(source_ids_, pids, x_2d, ax):
    pid_dict = {}
    size = 80
    id_counter = len(set(source_ids_.tolist()))
    #print('id_counter', id_counter)
    for i in range(id_counter):
        index = np.where(source_ids_==i)[0]
        ids = pids[index] #第i个 domain 中 pids
        
        pid_dict[i] = list(set(ids)) 
    for p in range(source_ids_.shape[0]):
        color = point_color[source_ids_[p]] #根据domain 分配color
        pid_list = pid_dict[source_ids_[p]]
        pid = pids[p]
        
        idx= pid_list.index(pid)
        #print(pid, idx)
        shape = point_shape[idx]
        x = x_2d[p,0]
        y = x_2d[p,1]

        ax.scatter(x, y, marker=shape, s=size, c=color, alpha=0.5)
    
    return 
def imscatter(x, y, im_paths, attn_maps, ax=None, zoom=1, attn=False):
    if ax is None:
        ax = plt.gca()

    x, y = np.atleast_1d(x, y)
    artists = []
    for i, (x0, y0) in enumerate(zip(x, y)):
        try:
            #image = plt.imread(image)
            image = Image.open(im_paths[i])
        except TypeError:
            # Likely already an array...
            pass
        #image = image.resize((40,80))
        #resized_image = image.resize((40,80), Image.BICUBIC)
        #image = np.array(resized_image, dtype='float32')
        image = image.resize((40,80))
        if attn:
            image = draw_heatmap(image, attn_maps[i])
        im = OffsetImage(image, zoom=zoom)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        print(x0,y0)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def draw_heatmap(img, attn):
    img = img.convert('RGB') 
    img = np.asarray(img)
    h, w, _ = img.shape
    attn = attn.squeeze()
    coff = attn.data.cpu().numpy()
    coff =  (coff - coff.min()) / (coff.max() - coff.min())
    coff = np.clip(coff, 0, 1)
    coff = np.uint8(255 * coff)
    heatmap = cv2.applyColorMap(cv2.resize(coff, (w, h)), cv2.COLORMAP_JET)
    print(attn.shape,coff.shape, heatmap.shape, img.shape)
    result = heatmap * 0.15 + img * 0.8
    #result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    print(result.shape)
    result=Image.fromarray(result.astype(np.uint8))
    return result

'''
import matplotlib.pyplot as plt
import numpy as np



fig, ax = plt.subplots()

# Define a 1st position to annotate (display it with a marker)
xy = (0.5, 0.7)
ax.plot(xy[0], xy[1], ".r")

# Annotate the 1st position with a text box ('Test 1')
offsetbox = TextArea("Test 1")

ab = AnnotationBbox(offsetbox, xy,
                    xybox=(-20, 40),
                    xycoords='data',
                    boxcoords="offset points",
                    arrowprops=dict(arrowstyle="->"))
ax.add_artist(ab)

# Annotate the 1st position with another text box ('Test')
offsetbox = TextArea("Test")

ab = AnnotationBbox(offsetbox, xy,
                    xybox=(1.02, xy[1]),
                    xycoords='data',
                    boxcoords=("axes fraction", "data"),
                    box_alignment=(0., 0.5),
                    arrowprops=dict(arrowstyle="->"))
ax.add_artist(ab)

# Define a 2nd position to annotate (don't display with a marker this time)
xy = [0.3, 0.55]

# Annotate the 2nd position with a circle patch
da = DrawingArea(20, 20, 0, 0)
p = Circle((10, 10), 10)
da.add_artist(p)

ab = AnnotationBbox(da, xy,
                    xybox=(1.02, xy[1]),
                    xycoords='data',
                    boxcoords=("axes fraction", "data"),
                    box_alignment=(0., 0.5),
                    arrowprops=dict(arrowstyle="->"))

ax.add_artist(ab)

# Annotate the 2nd position with an image (a generated array of pixels)
arr = np.arange(100).reshape((10, 10))
im = OffsetImage(arr, zoom=2)
im.image.axes = ax

ab = AnnotationBbox(im, xy,
                    xybox=(-50., 50.),
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0.3,
                    arrowprops=dict(arrowstyle="->"))

ax.add_artist(ab)

# Annotate the 2nd position with another image (a Grace Hopper portrait)
with get_sample_data("grace_hopper.jpg") as file:
    arr_img = plt.imread(file)

imagebox = OffsetImage(arr_img, zoom=0.2)
imagebox.image.axes = ax

ab = AnnotationBbox(imagebox, xy,
                    xybox=(120., -80.),
                    xycoords='data',
                    boxcoords="offset points",
                    pad=0.5,
                    arrowprops=dict(
                        arrowstyle="->",
                        connectionstyle="angle,angleA=0,angleB=90,rad=3")
                    )

ax.add_artist(ab)

# Fix the display limits to see everything
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

plt.show()
'''



















    

'''
# Load a numpy record array from yahoo csv data with fields date, open, close,
# volume, adj_close from the mpl-data/example directory. The record array
# stores the date as an np.datetime64 with a day unit ('D') in the date column.
price_data = (cbook.get_sample_data('goog.npz', np_load=True)['price_data']
              .view(np.recarray))
price_data = price_data[-250:]  # get the most recent 250 trading days

delta1 = np.diff(price_data.adj_close) / price_data.adj_close[:-1]

# Marker size in units of points^2
volume = (15 * price_data.volume[:-2] / price_data.volume[0])**2
close = 0.003 * price_data.close[:-2] / 0.003 * price_data.open[:-2]

fig, ax = plt.subplots()
ax.scatter(delta1[:-1], delta1[1:], c=close, s=volume, alpha=0.5)

ax.set_xlabel(r'$\Delta_i$', fontsize=15)
ax.set_ylabel(r'$\Delta_{i+1}$', fontsize=15)
ax.set_title('Volume and percent change')

ax.grid(True)
fig.tight_layout()

plt.show()
'''   
    
    
    
    
    
    
'''
feat_ = feat_.data.numpy()
x_2d = TSNE(perplexity=10).fit(feats_)


index_1 = np.where(source_ids==0)[0]
index_2 = np.where(source_ids==1)[0]
index_3 = np.where(source_ids==2)[0]
index_4 = np.where(source_ids==3)[0]
index_5 = np.where(source_ids==4)[0]


feat_1 = extracted_features[index_1]
feat_2 = extracted_features[index_2]
feat_3 = extracted_features[index_3]
feat_4 = extracted_features[index_4]
feat_5 = extracted_features[index_5]

if len(feat_1.shape)==4:
    N, C, H, W = feat_1.shape
    feat_1 = feat_1.view(N, C, -1).mean(2)
    feat_2 = feat_2.view(N, C, -1).mean(2)
    feat_3 = feat_3.view(N, C, -1).mean(2)
    feat_4 = feat_4.view(N, C, -1).mean(2)
    feat_5 = feat_5.view(N, C, -1).mean(2)

N = feat_1.shape[0]
feats_ = torch.cat([feat_1, feat_2, feat_3, feat_4, feat_5],dim=0).view(5*N,-1).data.numpy()

x_2d = TSNE(perplexity=10).fit(feats_)

for 


x_1 = x_2d[:length, 0]
y_1 = x_2d[:length, 1]
x_2 = x_2d[length:length*2, 0]
y_2 = x_2d[length:length*2, 1]

x_3 = x_2d[length*2:length*3, 0]
y_3 = x_2d[length*2:length*3, 1]

x_4 = x_2d[length*3:length*4, 0]
y_4 = x_2d[length*3:length*4, 1]

x_5 = x_2d[length*4:length*5, 0]
y_5 = x_2d[length*4:length*5, 1]



print('done1')
size = 1
plt.scatter(x_1, y_1, marker='o', s=size, c='r', label="market")
plt.scatter(x_2, y_2, marker='^', s=size, c='g', label="duke")
plt.scatter(x_3, y_3, marker='x', s=size, c='b', label="cuhk03")
plt.scatter(x_4, y_4, marker='*', s=size, c='y', label="sysu")
plt.scatter(x_5, y_5, marker='s', s=size, c='k', label="cuhk02")
print('done2')
#plt.scatter(x_3, y_3, marker='s', s=size, label="MSMT")
#plt.scatter(x_4, y_4, marker='D', s=size, label="Cuhk03")
plt.savefig('1.jpg')

print('done3')
plt.close(0) 
'''


    
    
    



